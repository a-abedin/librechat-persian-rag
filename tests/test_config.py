from src.config import AppConfig

def test_default_config_values():
    config = AppConfig()
    assert config.max_answer_words == 4
    assert config.temperature == 0.0
    assert config.collection_name == "librechat_knowledge_base"
