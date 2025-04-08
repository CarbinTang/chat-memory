class EnhancedMemoryManager(MemoryManager):
    def __init__(self, llm_api):
        super().__init__(llm_api)
        self.intent_pipeline = pipeline("text-classification", model="joeddav/xlm-roberta-large-xnli")
        self.entity_types = ["地址", "饮食偏好", "过敏原", "日程时间"]
    
    def _detect_intent(self, query):
        """识别查询意图"""
        intents = self.intent_pipeline(query, candidate_labels=["餐饮推荐", "日程查询", "旅行规划", "其他"])
        return intents['labels'][0]
    
    def _get_related_entities(self, dialog_id, intent):
        """根据意图获取相关实体"""
        entity_mapping = {
            "餐饮推荐": ["地址", "饮食偏好", "过敏原"],
            "旅行规划": ["地址", "日程时间"],
            "日程查询": ["日程时间"]
        }
        target_types = entity_mapping.get(intent, [])
        
        # 从关键记忆中检索相关实体
        filter_expr = f"dialog_id == '{dialog_id}' && type in {target_types}"
        results = self.key_collection.query(expr=filter_expr)
        return {ent['type']: ent['value'] for ent in results}
    
    def _enhance_query(self, original_query, entities):
        """用记忆实体增强查询"""
        context_str = " ".join([f"[{k}:{v}]" for k,v in entities.items()])
        return f"{original_query} {context_str}"
    
    def smart_search(self, dialog_id, query):
        """智能联想搜索"""
        # 步骤1：识别意图
        intent = self._detect_intent(query)
        
        # 步骤2：获取相关实体
        related_entities = self._get_related_entities(dialog_id, intent)
        
        # 步骤3：增强查询
        enhanced_query = self._enhance_query(query, related_entities)
        print(f"增强后查询: {enhanced_query}")
        
        # 步骤4：执行记忆检索
        memories = self.retrieve_memories(dialog_id, enhanced_query)
        
        # 步骤5：构建上下文
        context = {
            "intent": intent,
            "entities": related_entities,
            "memories": memories
        }
        return context

    def generate_response(self, dialog_id, query):
        """生成增强响应"""
        context = self.smart_search(dialog_id, query)
        
        # 构建系统提示
        prompt_template = """
        已知用户信息：
        {% for key,value in entities.items() %}
        - {{ key }}: {{ value }}
        {% endfor %}
        
        历史相关对话：
        {% for mem in memories %}
        [{{ mem.summary }}]
        {% endfor %}
        
        用户请求：{{ query }}
        请生成友好自然的回复：
        """
        
        return self.llm.generate(
            template=prompt_template,
            entities=context['entities'],
            memories=context['memories'][:3],
            query=query
        )

# 增强关键信息存储（修改MemoryEnhancer）
class EnhancedMemoryEnhancer(MemoryEnhancer):
    def extract_critical_info(self, text):
        entities = super().extract_critical_info(text)
        
        # 添加类型标签
        type_prompt = f"分类以下信息类型（{self.entity_types}）：{text}"
        entity_type = self.llm.generate(type_prompt).strip()
        
        return {
            **entities,
            "type": entity_type if entity_type in self.entity_types else "其他"
        }

# 使用示例
if __name__ == "__main__":
    manager = EnhancedMemoryManager(MockLLM())
    
    # 模拟对话
    dialog_id = "user_789"
    manager.add_memory(dialog_id, {
        "speaker": "user", 
        "text": "我住在北京市海淀区中关村"
    })
    manager.add_memory(dialog_id, {
        "speaker": "user",
        "text": "我对海鲜过敏，不能吃虾类"
    })
    
    # 执行智能查询
    query = "帮我找一家餐厅"
    response = manager.generate_response(dialog_id, query)
    print(f"智能回复：{response}")