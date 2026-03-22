import gradio as gr

# ============================================================
# 最小構成の設定管理クラス（GUI が要求する機能をすべて実装）
# ============================================================

class DummyConfigManager:
    def __init__(self):
        # グローバル設定
        self.settings = {
            "vram_gb": 8,
            "ram_gb": 16,
            "model": "wd14",
        }

        # モデルごとの設定（GUI が参照する）
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

    # main.py が要求
    def get_global_settings(self):
        return self.settings

    # main.py が要求
    def save_global_settings(self, new_settings: dict):
        self.settings.update(new_settings)

    # handlers.py が要求
    def get_model_config(self, model_id: str):
        return self.model_configs.get(model_id, {"description": "説明なし", "params": {}})


# ============================================================
# CaptioningApp（GUI が要求する属性をすべて実装）
# ============================================================

class CaptioningApp:
    def __init__(self):
        # 設定管理
        self.config_mgr = DummyConfigManager()

        # モデル一覧（GUI が参照）
        self.available_models = {
           
