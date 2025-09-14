import gradio as gr
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataroom.agent.agent import Agent
from dataroom.tools import RealEstateRAGTool
from dataroom.utils.utils import load_txt_prompts_from_file
from dataroom.rag.build_database import VectorDatabaseManager

class RealEstateInterface:
    """Simplified interface: single chat + file upload"""

    def __init__(self, agent: Optional[Agent] = None):
        self.agent = agent
        self.db_manager: Optional[VectorDatabaseManager] = None

    # ---- Core Functions ----
    def initialize_agent(self) -> bool:
        if self.agent is not None:
            return True
        try:
            prompts = load_txt_prompts_from_file("docs/system_prompt.txt")
            system_prompt = prompts["system_prompt"]
            rag_tool = RealEstateRAGTool()
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langgraph.checkpoint.memory import MemorySaver
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.7,
                top_p=0.95,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.agent = Agent(
                model=llm,
                tools=[rag_tool],
                system_prompt=system_prompt,
                checkpointer=MemorySaver(),
                log_tools=True,
                log_dir="logs"
            )
            return True
        except Exception as e:
            print(f"Init error: {e}")
            return False

    def chat_with_agent(self, message: str, history: List[List[str]]):
        if not self.agent:
            return "❌ Agent not initialized", history
        if not message.strip():
            return "", history
        try:
            history.append([message, "..."])
            response = self.agent.workflow.invoke(
                {"messages": [("user", message)]},
                config={"configurable": {"thread_id": "single_thread"}}
            )
            answer = response["messages"][-1].content
            history[-1][1] = answer
            return "", history
        except Exception as e:
            history[-1][1] = f"❌ Error: {e}"
            return "", history

    def upload_document(self, file) -> str:
        if not file:
            return "❌ No file selected"
        try:
            if self.agent is None:
                self.initialize_agent()  # Ensure tools available
            if self.db_manager is None:
                self.db_manager = VectorDatabaseManager()
            file_path = file.name
            suffix = Path(file_path).suffix.lower()
            if suffix == ".pdf":
                result = self.db_manager.add_pdf_document(file_path)
                if "error" in result:
                    return f"❌ PDF failed: {result['error']}"
                return f"✅ PDF added, pages: {result.get('pages_added', 0)}"
            elif suffix == ".csv":
                result = self.db_manager.add_csv_document(file_path)
                if "error" in result:
                    return f"❌ CSV failed: {result['error']}"
                return f"✅ CSV added, chunks: {result.get('chunks_added', 0)}"
            return f"❌ Unsupported file type: {suffix}"
        except Exception as e:
            return f"❌ Upload error: {e}"

    # ---- UI ----
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="🏠 Real Estate Dataroom",
            theme=gr.themes.Soft(),
            css="""
            .header {text-align:center;padding:18px;margin-bottom:10px;background:linear-gradient(90deg,#4e54c8,#8f94fb);color:#fff;border-radius:8px}
            """
        ) as demo:
            gr.HTML("""
            <div class='header'>
              <h1>🏠 Real Estate Dataroom</h1>
              <p>Upload a PDF / CSV then ask questions</p>
            </div>
            """)

            status = gr.Textbox(label="Status", value="� Initializing...", interactive=False)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 File Upload")
                    file_comp = gr.File(label="Select File (PDF / CSV)", file_count="single", file_types=[".pdf", ".csv"])
                    upload_btn = gr.Button("📤 Upload & Ingest")
                    upload_status = gr.Textbox(label="Result", interactive=False, lines=4)
                    list_btn = gr.Button("📂 List Current Files")
                    file_list = gr.Markdown(value="(Not queried yet)")
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="💬 Q&A", height=520)
                    with gr.Row():
                        user_input = gr.Textbox(placeholder="Enter your question...", scale=5)
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("🧹 Clear Chat", variant="secondary")

            # --- Event Functions ---
            def _init():
                ok = self.initialize_agent()
                return "✅ Ready" if ok else "❌ Initialization failed"

            def _send(msg, history):
                return self.chat_with_agent(msg, history)

            def _clear():
                return "", []

            def _upload(f):
                return self.upload_document(f)

            def _list_files():
                try:
                    if self.db_manager is None:
                        self.db_manager = VectorDatabaseManager()
                    data = self.db_manager.list_documents()
                    if "error" in data:
                        return f"❌ {data['error']}"
                    lines = ["### Current Database Files"]
                    lines.append(f"PDF files: {data['summary']['pdf_files']}  |  CSV files: {data['summary']['csv_files']}")
                    if data['summary']['pdf_files'] == 0 and data['summary']['csv_files'] == 0:
                        lines.append("\n_No documents have been ingested yet. Please upload a PDF or CSV first._")
                        return "\n".join(lines)
                    if data['pdf']:
                        lines.append("\n**PDF:**")
                        for fname, meta in data['pdf'].items():
                            pages = meta.get('total_pages')
                            lines.append(f"- {fname} (pages: {pages})")
                    if data['csv']:
                        lines.append("\n**CSV:**")
                        for fname, meta in data['csv'].items():
                            rows = meta.get('total_rows')
                            lines.append(f"- {fname} (rows: {rows})")
                    return "\n".join(lines)
                except Exception as e:
                    return f"❌ Failed to list files: {e}"

            # Event bindings
            demo.load(_init, outputs=[status])
            send_btn.click(_send, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
            user_input.submit(_send, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
            clear_btn.click(_clear, outputs=[user_input, chatbot])
            upload_btn.click(_upload, inputs=[file_comp], outputs=[upload_status])
            list_btn.click(_list_files, outputs=[file_list])

        return demo

def create_demo(agent: Optional[Agent] = None) -> gr.Blocks:
    """Create the demo interface"""
    interface = RealEstateInterface(agent)
    return interface.create_interface()
