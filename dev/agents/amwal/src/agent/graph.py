from typing import List, TypedDict, Callable, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
import json
from dotenv import load_dotenv
from langchain_core.callbacks.base import AsyncCallbackHandler

# تأكد من تحميل متغيرات البيئة لـ API Key الخاص بك
load_dotenv()

class FinancialAssistantState(TypedDict):
    messages: List[BaseMessage]
    parsed_message: str
    mock_data: Optional[dict]
    stream_callback: Optional[Callable[[str], None]]

# 🔸 تهيئة LLM (النموذج اللغوي)
llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True
)

class StreamCaptureHandler(AsyncCallbackHandler):
    def __init__(self, on_token):
        self.on_token = on_token
        print(f"StreamCaptureHandler: Initialized with on_token callback: {on_token}")

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when the LLM generates a new token"""
        print(f"StreamCaptureHandler: on_llm_new_token called. Token received: '{token[:10]}...'") # Print a snippet
        if self.on_token:
            await self.on_token(token)
        else:
            print("StreamCaptureHandler: Warning - on_token callback is None.")

# 🔸 العقدة 1: محلل المدخلات (Input Parser)
def parse_node(state: FinancialAssistantState) -> FinancialAssistantState:
    print("\n--- Entering parse_node ---")
    last_user = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
    message_text = last_user.content if last_user else ""

    # fallback
    message = message_text
    mock_data = state.get("mock_data")  # default from state

    try:
        parsed = json.loads(message_text)
        message = parsed.get("message", message_text)
        # Only override if mock_data was provided in the message JSON
        if parsed.get("mock_data") is not None:
            mock_data = parsed["mock_data"]
        print("Parser: Input was valid JSON. Extracted message and mock_data.")
    except json.JSONDecodeError:
        print("Parser: Input is not JSON, keeping upstream mock_data.")
    except Exception as e:
        print(f"Parser: Unexpected error: {e}, keeping upstream mock_data.")

    return {
        "messages": state["messages"],
        "parsed_message": message,
        "mock_data": mock_data,
        "stream_callback": state.get("stream_callback")
    }

async def abdurrahman_node(state: FinancialAssistantState) -> FinancialAssistantState:
    print("\n--- Entering abdurrahman_node ---")

    default_data = {
        "name": "عبدالله",
        "monthly_income": 9000,
        "financial_goal": "شراء سيارة جديدة خلال 6 أشهر",
        "current_savings": 5000,
        "monthly_commitments": 3200,
        "financial_discipline_score": "86%",
        "credit_score": "23%",
        "transactions": [
            {"date": "2025-07-01", "description": "راتب شهري", "amount": +9000},
            {"date": "2025-07-02", "description": "إيجار شقة", "amount": -2500},
            {"date": "2025-07-03", "description": "فواتير كهرباء وماء", "amount": -400},
            {"date": "2025-07-04", "description": "قهوة يومية", "amount": -45},
            {"date": "2025-07-06", "description": "مطعم", "amount": -120},
            {"date": "2025-07-08", "description": "ادخار تلقائي", "amount": -800}
        ]
    }

    mock_data = state.get("mock_data") or default_data
    print(f"Abdurrahman: Using mock data: {mock_data is not None}")

    sys_msg = SystemMessage(content=(
        """أنت عبدالرحمن، المستشار المالي الذكي داخل تطبيق أموال.
تم تطوير عبدالرحمن وتطبيق أموال بواسطة **عبدالرحمن حسين**: مؤسس GeniusAICo ومهندس تعلم آلة | مهندس أنظمة ذكاء اصطناعي، بهدف بناء حلول ذكية مؤتمتة لمستقبل الأعمال. هذا المشروع هو مشاركة في **هاكاثون أمد فنتك** الذي يقام بالشراكة بين الإنماء وأكاديمية طويق، ضمن مسار **التعليم المالي (التوعية)**، بهدف رفع الوعي المالي لدى الأفراد باستخدام أدوات مبتكرة لمفاهيم الادخار والميزانية والائتمان.

أنت تتحدث مع المستخدم بأسلوب ودود وشخصي **كأنك أخوه الكبير اللي يهمه مصلحته ويفهم عليه زين**، هدفك مساعدته يفهم وضعه المالي، ويحقق أهدافه بخطط عملية بسيطة **وبكلام يدخل القلب**.

تعليمات هامة لأسلوبك:
1. اجعل إجاباتك قصيرة وواضحة، وتكلم باللهجة السعودية **مثل ما يتكلمون الشباب**.
2. استخدم لهجة تشجيعية ومباشرة **مليانة حماس وإيجابية**.
3. اقترح نصائح عملية بناءً على بياناته الحقيقية فقط **شيء واقعي ومفيد له**.
4. إذا سأل عن شيء غير مالي، ذكّره بلطافة أن تركيزنا مالي بحت **وأن هنا تخصصنا الأساسي**.

📊 هذه بياناته الحالية:
"""
            f"{json.dumps(mock_data, ensure_ascii=False, indent=2)}\n\n"
            "اعتمد عليها للإجابة على أي سؤال يطرحه الآن."
    ))
    print(f"Abdurrahman: System message created.")

    messages_for_llm = [sys_msg] + state["messages"]
    print(f"Abdurrahman: Total messages for LLM: {len(messages_for_llm)}")
    if any(m.type == "human" for m in messages_for_llm):
        print(f"Abdurrahman: Last user message content: '{[m for m in messages_for_llm if m.type == 'human'][-1].content}'")

    print("Abdurrahman: Calling llm.ainvoke. Expecting auto-streaming if graph is streaming.")
    
    # ✅ التغيير الحاسم: استخراج دالة الـ callback من الـ state
    stream_callback_func = state.get("stream_callback")
    callbacks = []
    if stream_callback_func:
        # ✅ إنشاء نسخة من StreamCaptureHandler مع دالة الـ callback
        callbacks.append(StreamCaptureHandler(stream_callback_func))
        print("Abdurrahman: Attached StreamCaptureHandler to LLM call.")

    # ✅ تمرير الـ callbacks إلى llm.ainvoke باستخدام الوسيط 'config'
    response = await llm.ainvoke(messages_for_llm, config={"callbacks": callbacks})
    
    print(f"Abdurrahman: LLM ainvoke completed. Response content length: {len(response.content)}")
    
    # Return the updated state, appending the full AI message
    # and ensuring all other custom state fields are propagated.

    return {
        "messages": state["messages"] + [response],
        "parsed_message": state["parsed_message"],
        "mock_data": state["mock_data"],
        "stream_callback": state["stream_callback"] # KEEP propagating this, as main.py relies on it being in the state
    }


# 🔸 العقدة 3: المنهي (Finalizer) - No change needed here
def final_node(state: FinancialAssistantState) -> FinancialAssistantState:
    print("\n--- Entering final_node ---")
    print(f"Finalizer: Final messages count: {len(state['messages'])}")
    print(f"Finalizer: Last message type: {state['messages'][-1].type if state['messages'] else 'None'}")
    print(f"Finalizer: Stream callback still present: {state.get('stream_callback') is not None}")
    return state

# ... (rest of your graph.py file, including builder and graph compilation)

## **بناء وتجميع الجراف (الرسم البياني)**
builder = StateGraph(FinancialAssistantState)

builder.add_node("parser", parse_node)
builder.add_node("abdurrahman", abdurrahman_node)
builder.add_node("final", final_node)

builder.set_entry_point("parser")

builder.add_edge("parser", "abdurrahman")
builder.add_edge("abdurrahman", "final")

graph = builder.compile()