import gradio as gr

# ============================================================
# 最小構成の設定管理クラス（ダミー）
# ============================================================

class DummyConfigManager:
    def __init__(self):
        self.settings = {
            "vram_gb": 8,
            "ram_gb": 16,
            "model": "wd14",
        }

    def get_global_settings(self):
        return self.settings

    def save_global_settings(self, new_settings: dict):
        self.settings.update(new_settings)


# ============================================================
# CaptioningApp（config_mgr 付き）
# ============================================================

class CaptioningApp:
    def __init__(self):
        # main.py が必要とする属性
        self.config_mgr = DummyConfigManager()

    def generate_caption(self, image, model_name, prompt, negative_prompt):
        if image is None:
            return "画像が選択されていません。"

        base = f"[Model: {model_name}] "
        if prompt:
            base += f"[Prompt: {prompt}] "
        if negative_prompt:
            base += f"[Negative: {negative_prompt}] "

        eng = base + "A sample caption describing the image."
        jpn = "（日本語訳）画像を説明するサンプルキャプションです。"

        return f"【原文】\n{eng}\n\n【訳文】\n{jpn}"

    def create_ui(self):
        with gr.Blocks(title="A Thousand Words - 日本語版") as demo:
            gr.Markdown(
                """
# 🖼️ A Thousand Words（日本語版 GUI）

画像からキャプションを生成し、  
**上に英語（原文）、下に日本語訳（訳文）** を表示します。
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="入力画像", type="pil")

                    model_name = gr.Dropdown(
                        label="使用するモデル",
                        choices=["wd14", "joycaption", "florence2", "qwen-vl"],
                        value="wd14",
                    )

                    prompt = gr.Textbox(
                        label="追加プロンプト（任意）",
                        placeholder="例：構図やスタイルの指定など",
                        lines=2
                    )

                    negative_prompt = gr.Textbox(
                        label="ネガティブプロンプト（任意）",
                        placeholder="例：除外したい要素など",
                        lines=2
                    )

                    run_button = gr.Button("キャプション生成を開始")

                with gr.Column(scale=1):
                    gr.Markdown("### 📝 生成結果（原文＋訳文）")
                    output = gr.Textbox(label="キャプション", lines=10)

            run_button.click(
                fn=self.generate_caption,
                inputs=[image_input, model_name, prompt, negative_prompt],
                outputs=[output],
            )

        return demo
