from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import numpy as np

# 配置参数
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
DIMENSION = 384
MEMORY_COLLECTION_NAME = 'chat_memories'
MAX_WINDOW_SIZE = 5  # 最大窗口限制
MIN_SIMILARITY = 0.65  # 话题相似度阈值
SUMMARY_PROMPT = "请用1-2句话总结以下对话内容的核心信息，保留关键实体和意图：\n{context}"

class TopicDetector:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.window_embeddings = []

    def update_window(self, text):
        """更新窗口并返回是否需要切分"""
        new_embedding = self.encoder.encode([text])[0]
        
        if len(self.window_embeddings) > 0:
            similarities = [
                self._cos_sim(new_embedding, existing)
                for existing in self.window_embeddings
            ]
            avg_similarity = np.mean(similarities)
            if avg_similarity < MIN_SIMILARITY:
                return True
        
        self.window_embeddings.append(new_embedding)
        # 保持最近3轮对话的嵌入用于比较
        if len(self.window_embeddings) > 3:
            self.window_embeddings.pop(0)
        return False

    def _cos_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class MemoryManager:
    def __init__(self, llm_api):
        self.llm = llm_api
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.current_dialog = []
        self.topic_detector = TopicDetector()
        
        # 初始化Milvus连接
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        self._create_collection()

    def _create_collection(self):
        if not utility.has_collection(MEMORY_COLLECTION_NAME):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="dialog_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
                FieldSchema(name="content", dtype=DataType.JSON),
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="timestamp", dtype=DataType.INT64)
            ]
            schema = CollectionSchema(fields, description="Chat memory storage")
            self.collection = Collection(MEMORY_COLLECTION_NAME, schema)
            
            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            self.collection.create_index("vector", index_params)
        else:
            self.collection = Collection(MEMORY_COLLECTION_NAME)
            self.collection.load()

    def _generate_summary(self, context):
        """使用大模型生成对话摘要"""
        prompt = SUMMARY_PROMPT.format(context="\n".join(context))
        response = self.llm.generate(prompt)
        return response.strip()

    def _should_split_window(self, new_utterance):
        """判断是否需要切分窗口"""
        # 基于话题相似度检测
        text_content = f"{new_utterance['speaker']}: {new_utterance['text']}"
        topic_changed = self.topic_detector.update_window(text_content)
        
        # 强制分割条件
        if len(self.current_dialog) >= MAX_WINDOW_SIZE:
            return True
        return topic_changed

    def add_memory(self, dialog_id, new_utterance):
        """添加新的对话记录"""
        self.current_dialog.append(new_utterance)
        
        if self._should_split_window(new_utterance):
            self._save_memory_block(dialog_id)
            self.current_dialog = [new_utterance]  # 新话题保留当前语句作为起始

    def _save_memory_block(self, dialog_id):
        """保存当前窗口为记忆块"""
        context = "\n".join([f"{u['speaker']}: {u['text']}" for u in self.current_dialog])
        summary = self._generate_summary(context)
        
        # 生成整体向量
        vector = self.encoder.encode([context])[0].tolist()
        
        # 构建存储数据
        memory_data = {
            "dialog_id": dialog_id,
            "vector": vector,
            "content": {
                "dialog_window": self.current_dialog,
                "speakers": list(set(u['speaker'] for u in self.current_dialog))
            },
            "summary": summary,
            "timestamp": int(datetime.now().timestamp())
        }
        
        # 插入Milvus
        insert_data = [
            [memory_data["dialog_id"]],
            [memory_data["vector"]],
            [memory_data["content"]],
            [memory_data["summary"]],
            [memory_data["timestamp"]]
        ]
        self.collection.insert(insert_data)
        
        # 清空当前窗口
        self.current_dialog = []

    def retrieve_memories(self, dialog_id, query, top_k=5):
        """检索相关记忆"""
        # 将查询转换为向量
        query_vector = self.encoder.encode([query])[0].tolist()
        
        # 构建搜索条件
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # 执行搜索
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=f"dialog_id == '{dialog_id}'",  # 限制在当前对话
            output_fields=["content", "summary"]
        )
        
        # 解析结果
        memories = []
        for hits in results:
            for hit in hits:
                memories.append({
                    "score": hit.score,
                    "content": hit.entity.get("content"),
                    "summary": hit.entity.get("summary")
                })
        return memories

# 使用示例
if __name__ == "__main__":
    class MockLLM:
        def generate(self, prompt):
            return "示例摘要：用户咨询了旅行计划和日程安排相关问题"

    manager = MemoryManager(MockLLM())
    
    # 测试话题变化的对话流
    dialog_id = "dynamic_test"
    conversations = [
        {"speaker": "user", "text": "我想订去北京的机票"},
        {"speaker": "assistant", "text": "您计划什么时候出发？"},
        {"speaker": "user", "text": "下周一早上"},
        # 话题变化
        {"speaker": "user", "text": "另外我的手机套餐怎么升级？"},
        {"speaker": "assistant", "text": "您需要哪个运营商的套餐？"},
        {"speaker": "user", "text": "中国移动的5G套餐"}
    ]
    
    for idx, conv in enumerate(conversations):
        manager.add_memory(dialog_id, conv)
        if len(manager.current_dialog) == 0:
            print(f"第{idx+1}轮对话后生成记忆块")
    
    # 检索测试
    results = manager.retrieve_memories(dialog_id, "手机套餐相关咨询")
    print("动态窗口检索结果：", json.dumps(results, indent=2, ensure_ascii=False))