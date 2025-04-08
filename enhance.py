# 新增关键实体提取和记忆强化模块
from transformers import pipeline

class MemoryEnhancer:
    def __init__(self):
        self.ner = pipeline("ner", aggregation_strategy="simple")
        self.keywords_prompt = "提取以下文本中需要长期记忆的重要信息，包括：\n- 个人偏好\n- 特定数字\n- 重要约定\n- 特殊要求\n文本：{text}"
    
    def extract_critical_info(self, text):
        """提取需要强化记忆的关键信息"""
        # 实体识别
        entities = self.ner(text)
        important_entities = [
            e['word'] for e in entities 
            if e['entity_group'] in ['PER', 'ORG', 'LOC', 'DATE', 'TIME']
        ]
        
        # 使用LLM提取重要信息
        prompt = self.keywords_prompt.format(text=text)
        keywords = self.llm.generate(prompt).split("\n")
        
        return {
            "entities": list(set(important_entities)),
            "keywords": [k.strip() for k in keywords if k.strip()]
        }

class EnhancedMemoryManager(MemoryManager):
    def __init__(self, llm_api):
        super().__init__(llm_api)
        self.enhancer = MemoryEnhancer()
        self._create_key_memory_collection()
        
        # 关键记忆缓存（对话级）
        self.key_memory_cache = {}

    def _create_key_memory_collection(self):
        # 创建关键信息专用集合
        self.key_collection = Collection("key_memories", schema=KeyMemorySchema())
        # ...（类似主集合的初始化逻辑）...

    def _save_key_memory(self, dialog_id, utterance, info):
        """存储关键记忆"""
        memory_data = {
            "dialog_id": dialog_id,
            "content": {
                "original_text": utterance['text'],
                "speaker": utterance['speaker'],
                "entities": info['entities'],
                "keywords": info['keywords']
            },
            "timestamp": int(datetime.now().timestamp())
        }
        # 插入关键记忆集合...
        # 同时更新缓存
        if dialog_id not in self.key_memory_cache:
            self.key_memory_cache[dialog_id] = []
        self.key_memory_cache[dialog_id].append(memory_data)

    def add_memory(self, dialog_id, new_utterance):
        """增强的记忆存储流程"""
        # 原始存储逻辑
        super().add_memory(dialog_id, new_utterance)
        
        # 关键信息检测
        text = f"{new_utterance['speaker']}: {new_utterance['text']}"
        info = self.enhancer.extract_critical_info(text)
        
        if len(info['entities']) > 0 or len(info['keywords']) > 0:
            self._save_key_memory(dialog_id, new_utterance, info)
            print(f"检测到关键信息：{info}")

    def retrieve_context(self, dialog_id, query):
        """增强的记忆检索"""
        # 基础记忆检索
        base_memories = super().retrieve_memories(dialog_id, query)
        
        # 关键记忆检索（向量+关键词）
        key_results = self.key_collection.search(
            query_vector=encode(query),
            filter_=f"dialog_id == '{dialog_id}'",
            params={"k": 3}
        )
        
        # 缓存检索
        cached_keys = self.key_memory_cache.get(dialog_id, [])
        
        # 合并结果并去重
        return self._rank_memories(
            base_memories + key_results + cached_keys, 
            query
        )

    def _rank_memories(self, memories, query):
        """基于相关性排序记忆"""
        # 1. 关键记忆优先
        # 2. 包含更多匹配实体的优先
        # 3. 时间较新的优先
        # ...（实现排序逻辑）...

    def generate_response(self, query, dialog_id):
        """生成响应时自动注入关键信息"""
        context = self.retrieve_context(dialog_id, query)
        
        # 提取关键实体作为系统提示
        key_entities = set()
        for mem in context:
            if 'entities' in mem:
                key_entities.update(mem['entities'])
        
        system_prompt = f"""
        当前对话重要信息：
        {', '.join(key_entities) if key_entities else '无'}
        
        请根据以下上下文回答问题：
        {context}
        """
        
        return self.llm.generate(system_prompt + query)

# 使用示例
if __name__ == "__main__":
    manager = EnhancedMemoryManager(MockLLM())
    
    # 测试对话
    dialog_id = "user_456_session_1"
    conversations = [
        {"speaker": "user", "text": "我对花生过敏，请注意饮食安排"},
        {"speaker": "assistant", "text": "已记录您的过敏史，将避免含花生成分"},
        {"speaker": "user", "text": "我想预定明天北京到上海的航班"},
        {"speaker": "assistant", "text": "找到3个航班选项，您偏好早班还是晚班？"},
        {"speaker": "user", "text": "请推荐上海迪士尼附近的酒店"},
    ]
    
    for conv in conversations:
        manager.add_memory(dialog_id, conv)
    
    # 后续提问
    query = "我刚才说的饮食限制是什么？"
    response = manager.generate_response(query, dialog_id)
    print(f"系统回复：{response}")