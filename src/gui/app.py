import gradio as gr

# ============================================================
# 設定管理（GUI が要求する機能をすべて実装）
# ============================================================

class DummyConfigManager:
    def __init__(self):
        # グローバル設定
        self.settings = {
            "vram_gb": 8,
            "ram_gb": 16,
            "model": "wd14",
        }

        # モデルごとの設定（GUI が参照）
        self.model_configs = {
            "wd14": {
                "description": "WD14 タグ付けモデル",
                "params": {}
            },
            "joycaption": {
                "description": "JoyCaption キャプションモデル",
                "params": {}
            },
            "florence2": {
                "description": "Florence2 VLM モデル",
                "params": {}
            },
            "qwen-vl": {
                "description": "Qwen-VL マルチモーダルモデル",
                "params": {}
            },
        }

        # マルチモデル設定（GUI が要求）
        self.multi_model_settings = {
            "enabled_models": ["wd14", "joycaption", "florence2", "qwen-vl"],
            "model_order": ["wd14", "joycaption", "florence2", "qwen-vl"],
        }

    # main.py が要求
    def get_global_settings(self):
        return self.settings

    # main.py が要求
    def save_global_settings(self, new_settings: dict):
        self.settings.update(new_settings)

    # handlers.py が要求
    def get_model_config(self, model_id: str):
        return self.model_configs.get(model_id, {"description": "説明なし", "params": {}})

    # multi_model_tab.py が要求
    def load_multi_model_settings(self):
        return self.multi_model_settings

    # multi_model_tab.py が要求
    def save_multi_model_settings(self, settings: dict):
        self.multi_model_settings.update(settings)


# ============================================================
# CaptioningApp（GUI が要求する属性をすべて実装）
# ============================================================

class CaptioningApp:
    def __init__(self):
        # 設定管理
        self.config_mgr = DummyConfigManager()

        # モデル一覧（GUI が参照）
        self.available_models = {
            "wd14": "WD14 タグ付けモデル",
            "joycaption": "JoyCaption キャプションモデル",
            "florence2": "Florence2 VLM",
            "qwen-vl": "Qwen-VL マルチモーダルモデル",
        }

        # GUI が要求するモデルリスト
        self.models = list(self.available_models.keys())

        # GUI が要求する「有効モデル」
        self.enabled_models = list(self.available_models.keys())

        # 初期モデル
        self.current_model_id = "wd14"

    # handlers.py が要求
    def get_model_description_html(self, model_id: str) -> str:
        cfg = self.config_mgr.get_model
