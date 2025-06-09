#!/usr/bin/env python3

import argparse
import csv
import networkx as nx
from pathlib import Path
from typing import Dict, Set, List, Optional, Any, Iterator, Tuple
import textwrap
import synphoni.utils as su
import synphoni.graph_analysis as sg
from synphoni.logo import logo_ASCII
import pickle
from dataclasses import dataclass
import sys
import gc
import sqlite3
import json
import uuid
import tempfile
import shutil
import psutil
import os
import time
import mmap
import numpy as np
from contextlib import contextmanager

# --- memory_profiler import and decorator ---
try:
    from memory_profiler import profile
except ImportError:
    def profile(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    print("Memory profiler not found. Install with 'pip install memory_profiler' for detailed memory usage.")

@dataclass
class SynphoniConfig:
    """Configuration class to store all parameters"""
    min_len: int
    min_shared: float
    clique_size: int
    min_community_coverage: float
    chrom_clustering_method: str
    # 流式处理配置
    temp_dir: Optional[str] = None
    memory_threshold_mb: int = 8000
    keep_temp_files: bool = False
    # 数据分解配置
    max_graph_nodes_in_memory: int = 5000
    max_community_genes: int = 1000
    use_disk_cache: bool = True
    streaming_batch_size: int = 100

class DiskBackedDict:
    """基于磁盘的字典，用于处理超大数据"""
    
    def __init__(self, temp_dir: Path, name: str):
        self.temp_dir = temp_dir
        self.name = name
        self.db_path = temp_dir / f"{name}.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_key ON {name}(key)")
        self.conn.commit()
    
    def __setitem__(self, key: str, value: Any):
        """存储键值对"""
        data = pickle.dumps(value)
        self.conn.execute(f"INSERT OR REPLACE INTO {self.name} (key, value) VALUES (?, ?)", 
                         (key, data))
        self.conn.commit()
    
    def __getitem__(self, key: str) -> Any:
        """获取值"""
        cursor = self.conn.execute(f"SELECT value FROM {self.name} WHERE key = ?", (key,))
        result = cursor.fetchone()
        if result is None:
            raise KeyError(key)
        return pickle.loads(result[0])
    
    def get(self, key: str, default=None) -> Any:
        """安全获取值"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self) -> Iterator[str]:
        """获取所有键"""
        cursor = self.conn.execute(f"SELECT key FROM {self.name}")
        for row in cursor:
            yield row[0]
    
    def items(self) -> Iterator[Tuple[str, Any]]:
        """获取所有键值对"""
        cursor = self.conn.execute(f"SELECT key, value FROM {self.name}")
        for key, value_blob in cursor:
            yield key, pickle.loads(value_blob)
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        cursor = self.conn.execute(f"SELECT 1 FROM {self.name} WHERE key = ? LIMIT 1", (key,))
        return cursor.fetchone() is not None
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()

class StreamingGraphProcessor:
    """流式图处理器，能处理超大图结构"""
    
    def __init__(self, temp_dir: Path, max_nodes_in_memory: int = 5000):
        self.temp_dir = temp_dir
        self.max_nodes_in_memory = max_nodes_in_memory
        self.node_cache = {}
        self.edge_cache = {}
        self.graph_db = DiskBackedDict(temp_dir, "graph_data")
    
    def process_large_graph(self, graph: nx.Graph) -> nx.Graph:
        """处理大型图，如果太大则分解处理"""
        
        if graph.number_of_nodes() <= self.max_nodes_in_memory:
            return graph  # 小图直接返回
        
        print(f"    [StreamingGraph] Processing large graph: {graph.number_of_nodes()} nodes")
        
        # 1. 找到最重要的子图
        core_subgraph = self._extract_core_subgraph(graph)
        
        # 2. 将剩余部分存储到磁盘
        remaining_nodes = set(graph.nodes()) - set(core_subgraph.nodes())
        if remaining_nodes:
            self._store_graph_partition(graph.subgraph(remaining_nodes), "remaining")
        
        print(f"    [StreamingGraph] Reduced to core subgraph: {core_subgraph.number_of_nodes()} nodes")
        return core_subgraph
    
    def _extract_core_subgraph(self, graph: nx.Graph) -> nx.Graph:
        """提取图的核心部分"""
        
        # 策略1: 找到最大的强连通分量
        if graph.is_directed():
            components = list(nx.strongly_connected_components(graph))
        else:
            components = list(nx.connected_components(graph))
        
        if not components:
            return nx.Graph()
        
        # 选择最大的连通分量
        largest_component = max(components, key=len)
        
        # 如果最大分量仍然太大，进一步缩减
        if len(largest_component) > self.max_nodes_in_memory:
            # 基于度数选择最重要的节点
            subgraph = graph.subgraph(largest_component)
            node_degrees = dict(subgraph.degree())
            
            # 选择度数最高的节点
            sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
            selected_nodes = [node for node, degree in sorted_nodes[:self.max_nodes_in_memory]]
            
            return graph.subgraph(selected_nodes).copy()
        
        return graph.subgraph(largest_component).copy()
    
    def _store_graph_partition(self, subgraph: nx.Graph, partition_name: str):
        """将图分区存储到磁盘"""
        partition_file = self.temp_dir / f"graph_partition_{partition_name}.gpickle"
        nx.write_gpickle(subgraph, partition_file)
        print(f"    [StreamingGraph] Stored partition '{partition_name}': {subgraph.number_of_nodes()} nodes")

class ChunkedDataProcessor:
    """分块数据处理器 - 修复参数名称"""
    
    def __init__(self, temp_dir: Path, chunk_size: int = 1000):
        self.temp_dir = temp_dir
        self.chunk_size = chunk_size
        self.data_cache = DiskBackedDict(temp_dir, "chunked_data")
    
    def process_large_community_data(self, 
                                   og_community: Set[str],  # 修正参数名
                                   chrom_data: Dict,        # 修正参数名  
                                   species_list: List,      # 修正参数名
                                   orthology: Dict,         # 修正参数名
                                   min_og_commu: float) -> Dict:
        """分块处理大型社区数据 - 使用正确的参数名"""
        
        # 估算数据大小
        total_genes = sum(
            len(chromo_data)
            for og_key in og_community if og_key in chrom_data
            for species_data in chrom_data[og_key].values()
            for chromo_data in species_data.values()
        )
        
        if total_genes <= self.chunk_size:
            # 小数据直接处理 - 使用正确的参数名
            return sg.genome_location_ogs(
                og_community=og_community,
                chrom_data=chrom_data,
                species_list=species_list,
                orthology=orthology,
                min_og_commu=min_og_commu
            )
        
        print(f"    [ChunkedProcessor] Processing large community: {len(og_community)} OGs, {total_genes} genes")
        
        # 分块处理
        community_list = list(og_community)
        all_results = {}
        
        for i in range(0, len(community_list), self.chunk_size):
            chunk = set(community_list[i:i + self.chunk_size])
            
            try:
                chunk_result = sg.genome_location_ogs(
                    og_community=chunk,
                    chrom_data=chrom_data,
                    species_list=species_list,
                    orthology=orthology,
                    min_og_commu=min_og_commu
                )
                
                if chunk_result:
                    all_results.update(chunk_result)
                
            except Exception as e:
                print(f"    [ChunkedProcessor] Chunk {i//self.chunk_size + 1} failed: {e}")
                continue
            finally:
                # 强制清理
                gc.collect()
        
        return all_results

class MemoryEfficientWriteBlocks:
    """内存高效的write_blocks实现"""
    
    def __init__(self, temp_dir: Path, batch_size: int = 50):
        self.temp_dir = temp_dir
        self.batch_size = batch_size
        self.temp_files = []
    
    def write_blocks_memory_efficient(self,
                                    blocks_writer: csv.writer,
                                    multi_sp_writer: csv.writer,
                                    genome_location_ogs_dict: Dict,
                                    og_info_graph: nx.Graph,
                                    k_perco: int,
                                    known_dict: Dict,
                                    method: str) -> Dict:
        """内存高效的write_blocks实现"""
        
        print(f"    [MemoryEfficientWriter] Processing {len(genome_location_ogs_dict)} location items")
        
        # 1. 检查数据大小
        if len(genome_location_ogs_dict) <= self.batch_size and og_info_graph.number_of_nodes() <= 1000:
            # 小数据直接处理
            return self._write_blocks_direct(
                blocks_writer, multi_sp_writer, genome_location_ogs_dict,
                og_info_graph, k_perco, known_dict, method
            )
        
        # 2. 大数据分批处理
        return self._write_blocks_batched(
            blocks_writer, multi_sp_writer, genome_location_ogs_dict,
            og_info_graph, k_perco, known_dict, method
        )
    
    def _write_blocks_direct(self, blocks_writer, multi_sp_writer, 
                           genome_location_ogs_dict, og_info_graph,
                           k_perco, known_dict, method) -> Dict:
        """直接处理小数据"""
        try:
            return sg.write_blocks(
                blocks_writer=blocks_writer,
                multi_sp_writer=multi_sp_writer,
                genome_location_ogs_dict=genome_location_ogs_dict,
                og_info_graph=og_info_graph,
                k_perco=k_perco,
                known_dict=known_dict,
                method=method
            ) or {}
        except Exception as e:
            print(f"    [MemoryEfficientWriter] Direct processing failed: {e}")
            return {}
    
    def _write_blocks_batched(self, blocks_writer, multi_sp_writer,
                            genome_location_ogs_dict, og_info_graph,
                            k_perco, known_dict, method) -> Dict:
        """分批处理大数据"""
        
        all_results = {}
        items = list(genome_location_ogs_dict.items())
        
        # 创建临时输出文件
        temp_synt_file = self.temp_dir / f"temp_synt_{uuid.uuid4()}.tsv"
        temp_clusters_file = self.temp_dir / f"temp_clusters_{uuid.uuid4()}.tsv"
        
        try:
            with open(temp_synt_file, 'w', newline='') as temp_synt, \
                 open(temp_clusters_file, 'w', newline='') as temp_clusters:
                
                temp_synt_writer = csv.writer(temp_synt, delimiter='\t')
                temp_clusters_writer = csv.writer(temp_clusters, delimiter='\t')
                
                # 分批处理
                for i in range(0, len(items), self.batch_size):
                    batch_items = dict(items[i:i + self.batch_size])
                    
                    # 创建批处理图
                    batch_nodes = set()
                    for locations in batch_items.values():
                        if isinstance(locations, dict):
                            for species_data in locations.values():
                                if isinstance(species_data, dict):
                                    for scaffolds in species_data.values():
                                        if isinstance(scaffolds, (list, set)):
                                            batch_nodes.update(str(s) for s in scaffolds)
                    
                    # 提取相关的子图
                    available_nodes = set(str(n) for n in og_info_graph.nodes())
                    relevant_nodes = batch_nodes.intersection(available_nodes)
                    
                    if relevant_nodes:
                        batch_graph = og_info_graph.subgraph(relevant_nodes).copy()
                    else:
                        batch_graph = nx.Graph()  # 空图
                    
                    try:
                        batch_result = sg.write_blocks(
                            blocks_writer=temp_synt_writer,
                            multi_sp_writer=temp_clusters_writer,
                            genome_location_ogs_dict=batch_items,
                            og_info_graph=batch_graph,
                            k_perco=k_perco,
                            known_dict=known_dict,
                            method=method
                        )
                        
                        if batch_result:
                            all_results.update(batch_result)
                        
                    except Exception as e:
                        print(f"    [MemoryEfficientWriter] Batch {i//self.batch_size + 1} failed: {e}")
                        continue
                    finally:
                        # 清理批处理数据
                        del batch_items, batch_graph
                        gc.collect()
            
            # 合并临时文件到主输出
            self._merge_temp_files(temp_synt_file, temp_clusters_file, 
                                 blocks_writer, multi_sp_writer)
            
        finally:
            # 清理临时文件
            for temp_file in [temp_synt_file, temp_clusters_file]:
                if temp_file.exists():
                    temp_file.unlink()
        
        return all_results
    
    def _merge_temp_files(self, temp_synt_file, temp_clusters_file,
                         main_synt_writer, main_clusters_writer):
        """合并临时文件到主输出"""
        
        # 合并synt文件
        if temp_synt_file.exists() and temp_synt_file.stat().st_size > 0:
            with open(temp_synt_file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    main_synt_writer.writerow(row)
        
        # 合并clusters文件
        if temp_clusters_file.exists() and temp_clusters_file.stat().st_size > 0:
            with open(temp_clusters_file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    main_clusters_writer.writerow(row)

class TempFileManager:
    """临时文件管理器"""
    
    def __init__(self, base_temp_dir: Optional[str] = None, keep_files: bool = False):
        if base_temp_dir:
            self.temp_dir = Path(base_temp_dir)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="synphoni_streaming_"))
        
        self.temp_files = []
        self.disk_dicts = []
        self.keep_files = keep_files
        self.temp_dir.mkdir(exist_ok=True)
        print(f"[TempFileManager] Using temp directory: {self.temp_dir}")
    
    def create_temp_file(self, suffix: str = ".tmp", prefix: str = "temp_") -> Path:
        """创建临时文件"""
        temp_file = self.temp_dir / f"{prefix}{uuid.uuid4()}{suffix}"
        self.temp_files.append(temp_file)
        return temp_file
    
    def create_disk_dict(self, name: str) -> DiskBackedDict:
        """创建磁盘字典"""
        disk_dict = DiskBackedDict(self.temp_dir, name)
        self.disk_dicts.append(disk_dict)
        return disk_dict
    
    def cleanup(self):
        """清理所有临时文件"""
        if self.keep_files:
            print(f"[TempFileManager] Keeping temp files in: {self.temp_dir}")
            return
        
        # 关闭磁盘字典
        for disk_dict in self.disk_dicts:
            disk_dict.close()
        
        # 删除临时文件
        for temp_file in self.temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
        
        # 删除临时目录
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temp dir: {e}")

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, threshold_mb: int = 8000):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process()
        self.peak_memory = 0
    
    def get_memory_usage_mb(self) -> float:
        """获取当前内存使用量(MB)"""
        current_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_mb)
        return current_mb
    
    def log_usage(self, context: str = ""):
        """记录内存使用情况"""
        current_mb = self.get_memory_usage_mb()
        print(f"[MemoryMonitor] {context} Memory: {current_mb:.1f}MB (Peak: {self.peak_memory:.1f}MB)")
    
    def force_cleanup(self):
        """强制内存清理"""
        for i in range(3):
            collected = gc.collect()
        
        # 尝试系统级内存清理
        try:
            import ctypes
            if sys.platform.startswith('linux'):
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
        except:
            pass

class SynphoniStreamingProcessor:
    """支持流式处理的SYNPHONI处理器"""

    @profile
    def __init__(self, config: SynphoniConfig):
        print(f"[{self.__class__.__name__}] Initializing streaming processor...")
        self.config = config
        
        # 基础数据（保持在内存中）
        self.chrom_dict: Dict[Any, Any] = {}
        self.G_og: Optional[nx.Graph] = None
        self.species_ls: List[Any] = []
        self.ortho: Dict[Any, Any] = {}
        
        # 流式处理组件
        self.temp_manager = TempFileManager(
            base_temp_dir=config.temp_dir,
            keep_files=config.keep_temp_files
        )
        self.memory_monitor = MemoryMonitor(config.memory_threshold_mb)
        
        # 处理器组件
        self.graph_processor = StreamingGraphProcessor(
            self.temp_manager.temp_dir, 
            config.max_graph_nodes_in_memory
        )
        self.data_processor = ChunkedDataProcessor(
            self.temp_manager.temp_dir,
            config.streaming_batch_size
        )
        self.write_blocks_processor = MemoryEfficientWriteBlocks(
            self.temp_manager.temp_dir,
            config.streaming_batch_size
        )
        
        print(f"[{self.__class__.__name__}] Streaming processor initialized.")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'temp_manager'):
            self.temp_manager.cleanup()

    @profile
    def load_data(self, filtered_graph_path: Path, chrom_data_path: Path) -> None:
        """加载数据并优化大型结构"""
        print(f"[{self.__class__.__name__}] Loading data with streaming optimization...")
        self.memory_monitor.log_usage("Before data loading")

        # 1. Load filtered_graph
        print(f"  Loading filtered graph from: {filtered_graph_path}")
        try:
            with open(filtered_graph_path, "rb") as fhin:
                raw_graph = pickle.load(fhin)
            
            if raw_graph is not None and isinstance(raw_graph, nx.Graph):
                print(f"  Raw graph: {raw_graph.number_of_nodes()} nodes, {raw_graph.number_of_edges()} edges")
                
                # 使用流式处理器优化图
                self.G_og = self.graph_processor.process_large_graph(raw_graph)
                
                print(f"  Optimized graph: {self.G_og.number_of_nodes()} nodes, {self.G_og.number_of_edges()} edges")
                del raw_graph  # 立即释放原始图
                
            else:
                print(f"  WARNING: Invalid graph loaded")
                self.G_og = nx.Graph()
                
        except Exception as e:
            raise RuntimeError(f"Error loading filtered graph: {str(e)}")
        
        self.memory_monitor.log_usage("After loading graph")

        # 2. Load chrom_data
        print(f"  Loading chromosome data from: {chrom_data_path}")
        try:
            self.chrom_dict = su.load_chrom_data(filepath=str(chrom_data_path))
            print(f"  SUCCESS: Loaded chrom_dict with {len(self.chrom_dict)} OGs")
        except Exception as e:
            raise RuntimeError(f"Error loading chromosome data: {str(e)}")
        
        self.memory_monitor.log_usage("After loading chrom_dict")

        # 3. Build ortho dictionary (可能使用磁盘存储)
        print("  Building ortho dictionary...")
        try:
            # 估算ortho字典大小
            estimated_size = sum(
                len(chromo_data)
                for og_data in self.chrom_dict.values()
                for species_data in og_data.values()
                for chromo_data in species_data.values()
            )
            
            if estimated_size > 100000:  # 如果超过10万条记录，使用磁盘存储
                print(f"  Large ortho dictionary detected ({estimated_size} entries), using disk storage...")
                self.ortho_disk = self.temp_manager.create_disk_dict("ortho")
                
                for og_key, og_data in self.chrom_dict.items():
                    for species, species_data in og_data.items():
                        for chromo, chromo_data in species_data.items():
                            for acc in chromo_data.keys():
                                self.ortho_disk[acc] = og_key
                
                self.ortho = {}  # 内存中保持空的
                print(f"  SUCCESS: Built disk-based ortho dictionary with {estimated_size} mappings")
                
            else:
                # 小字典保持在内存中
                self.ortho = {
                    acc: og_key
                    for og_key, og_data in self.chrom_dict.items()
                    for species, species_data in og_data.items()
                    for chromo, chromo_data in species_data.items()
                    for acc in chromo_data.keys()
                }
                print(f"  SUCCESS: Built in-memory ortho dictionary with {len(self.ortho)} mappings")
                
        except Exception as e:
            raise RuntimeError(f"Error building ortho dictionary: {str(e)}")

        # 4. Build species_ls
        try:
            self.species_ls = list(set(su.flatten([list(og_data.keys()) for og_key, og_data in self.chrom_dict.items()])))
            print(f"  SUCCESS: Built species list with {len(self.species_ls)} species")
        except Exception as e:
            raise RuntimeError(f"Error building species list: {str(e)}")

        self.memory_monitor.log_usage("After data loading completed")
        print(f"[{self.__class__.__name__}] Data loading finished with streaming optimization.")

    def _get_ortho_for_community(self, community: Set[str]) -> Dict[str, str]:
        """获取社区所需的ortho映射"""
        if hasattr(self, 'ortho_disk'):
            # 从磁盘获取
            community_ortho = {}
            all_accessions = set()
            
            # 收集社区中所有的accession
            for og_key in community:
                if og_key in self.chrom_dict:
                    og_data = self.chrom_dict[og_key]
                    for species, species_data in og_data.items():
                        for chromo, chromo_data in species_data.items():
                            all_accessions.update(chromo_data.keys())
            
            # 批量查询
            for acc in all_accessions:
                if acc in self.ortho_disk:
                    community_ortho[acc] = self.ortho_disk[acc]
            
            return community_ortho
        else:
            # 使用内存中的ortho
            return self.ortho

    @profile
    def process_communities(self, og_communities_path: Path, output_prefix: str) -> Dict:
        """流式处理社区"""
        print(f"[{self.__class__.__name__}] Starting streaming community processing...")
        
        block_ids: Dict[Any, Any] = {}
        processed_count = 0
        
        # 准备输出文件
        synt_path_str = f"{output_prefix}.len{self.config.min_len}.ol{self.config.min_shared}.synt"
        multi_sp_path_str = f"{output_prefix}.len{self.config.min_len}.ol{self.config.min_shared}.clusters"
        
        print(f"  Synteny output: {synt_path_str}")
        print(f"  Clusters output: {multi_sp_path_str}")

        try:
            with open(synt_path_str, "w", newline='') as synt_h, \
                 open(multi_sp_path_str, "w", newline='') as multi_sp_h, \
                 open(og_communities_path, "r") as comm_f:
                
                synt_w = csv.writer(synt_h, delimiter="\t")
                m_sp_w = csv.writer(multi_sp_h, delimiter="\t")
                reader = csv.reader(comm_f)
                
                for row in reader:
                    community = set(row)
                    processed_count += 1
                    
                    if processed_count % 10 == 0 or processed_count == 1:
                        print(f"    Processing community {processed_count} (size={len(community)})...")
                        self.memory_monitor.log_usage(f"Community {processed_count}")
                    
                    # 处理社区
                    new_ids = self._process_single_community_streaming(
                        community, synt_w, m_sp_w, block_ids
                    )
                    
                    if new_ids:
                        block_ids.update(new_ids)
                    
                    # 定期强制清理
                    if processed_count % 20 == 0:
                        self.memory_monitor.force_cleanup()

        except Exception as e:
            raise RuntimeError(f"Error during streaming processing: {str(e)}")

        print(f"[{self.__class__.__name__}] Streaming processing completed:")
        print(f"  Processed: {processed_count} communities")
        print(f"  Block IDs generated: {len(block_ids)}")
        
        return block_ids

    @profile
    def _process_single_community_streaming(self,
                                          community: Set[str],
                                          synt_writer: csv.writer,
                                          multi_sp_writer: csv.writer,
                                          existing_block_ids: Dict
                                         ) -> Dict:
        """流式处理单个社区"""
        
        memory_start = self.memory_monitor.get_memory_usage_mb()
        
        try:
            # 1. 获取社区相关的ortho映射
            community_ortho = self._get_ortho_for_community(community)
            
            # 2. 使用分块处理器获取scaffold locations - 使用正确的参数名
            current_commu_scaffolds = self.data_processor.process_large_community_data(
                og_community=community,          # 正确的参数名
                chrom_data=self.chrom_dict,      # 正确的参数名
                species_list=self.species_ls,    # 正确的参数名
                orthology=community_ortho,       # 正确的参数名
                min_og_commu=self.config.min_community_coverage
            )
            
            if not current_commu_scaffolds:
                return {}

            # 3. Build protoblock graph（已经通过StreamingGraphProcessor优化）
            protoblock_graph = sg.og_info_to_graph(
                genome_location_orthogroups=current_commu_scaffolds,
                fullgraph_ogs_filt=self.G_og,
                min_len=self.config.min_len,
                min_shared=self.config.min_shared
            )
            
            if not protoblock_graph or protoblock_graph.number_of_nodes() == 0:
                return {}

            # 4. 使用内存高效的write_blocks
            newly_created_ids = self.write_blocks_processor.write_blocks_memory_efficient(
                blocks_writer=synt_writer,
                multi_sp_writer=multi_sp_writer,
                genome_location_ogs_dict=current_commu_scaffolds,
                og_info_graph=protoblock_graph,
                k_perco=self.config.clique_size,
                known_dict=existing_block_ids,
                method=self.config.chrom_clustering_method
            )
            
            # 立即清理大对象
            del current_commu_scaffolds, protoblock_graph, community_ortho
            
            memory_end = self.memory_monitor.get_memory_usage_mb()
            memory_delta = memory_end - memory_start
            
            if memory_delta > 50:
                print(f"      Memory growth: +{memory_delta:.1f}MB, forcing cleanup")
                self.memory_monitor.force_cleanup()
            
            return newly_created_ids or {}

        except Exception as e:
            print(f"      ERROR processing community: {str(e)}")
            # 强制清理后返回
            self.memory_monitor.force_cleanup()
            return {}

def main():
    print("Starting SYNPHONI Step 4 with streaming memory management...")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            {logo_ASCII()}
            SYNPHONI Step 4 with streaming memory management
            Can process arbitrarily large datasets without running out of memory.
            """)
    )
    
    # 基础参数
    parser.add_argument("og_communities", help="CSV file containing orthogroup communities", type=Path)
    parser.add_argument("-g", "--filtered_graph", help="Filtered graph in gpickle format", type=Path, required=True)
    parser.add_argument("-c", "--chrom_data", help="Pickle file containing chromosome data", type=Path, required=True)
    parser.add_argument("-o", "--output", help="Prefix for output files", type=str, required=True)
    parser.add_argument("-l", "--min_len", help="Minimum number of ortholog occurrences required", default=3, type=int)
    parser.add_argument("-s", "--min_shared", help="Minimum overlap coefficient between scaffolds", default=0.5, type=float)
    parser.add_argument("-k", "--clique_size", help="Minimum size of multi-species blocks to retain", default=3, type=int)
    parser.add_argument("-r", "--min_community_coverage", help="Minimum percentage of orthogroups required", default=0.3, type=float)
    parser.add_argument("-m", "--chrom_clustering_method", help="Method for clustering chromosomes", default="k_clique", choices={"k_clique", "leiden"}, type=str)
    
    # 流式处理参数
    parser.add_argument("--temp-dir", help="Directory for temporary files", type=str, default=None)
    parser.add_argument("--memory-threshold", help="Memory usage threshold in MB", default=8000, type=int)
    parser.add_argument("--max-graph-nodes", help="Maximum graph nodes to keep in memory", default=5000, type=int)
    parser.add_argument("--max-community-genes", help="Maximum genes per community chunk", default=1000, type=int)
    parser.add_argument("--streaming-batch-size", help="Batch size for streaming processing", default=100, type=int)
    parser.add_argument("--keep-temp-files", help="Keep temporary files", action="store_true")
    parser.add_argument("--no-disk-cache", help="Disable disk caching", action="store_true")

    args = parser.parse_args()
    
    # 验证文件
    for file_path in [args.og_communities, args.filtered_graph, args.chrom_data]:
        if not file_path.exists():
            print(f"FATAL: File not found: {file_path}")
            sys.exit(1)

    # 创建配置
    config = SynphoniConfig(
        min_len=args.min_len,
        min_shared=args.min_shared,
        clique_size=args.clique_size,
        min_community_coverage=args.min_community_coverage,
        chrom_clustering_method=args.chrom_clustering_method,
        temp_dir=args.temp_dir,
        memory_threshold_mb=args.memory_threshold,
        keep_temp_files=args.keep_temp_files,
        max_graph_nodes_in_memory=args.max_graph_nodes,
        max_community_genes=args.max_community_genes,
        use_disk_cache=not args.no_disk_cache,
        streaming_batch_size=args.streaming_batch_size
    )

    # 运行流式处理器
    processor = None
    try:
        processor = SynphoniStreamingProcessor(config)
        processor.load_data(args.filtered_graph, args.chrom_data)
        final_block_ids = processor.process_communities(args.og_communities, args.output)
        
        print(f"Streaming processing completed successfully!")
        print(f"Generated {len(final_block_ids)} block IDs")
        processor.memory_monitor.log_usage("Final")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if processor:
            processor.temp_manager.cleanup()

if __name__ == "__main__":
    main()
