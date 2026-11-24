# Flask example
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "OK"
import os
import gradio as gr

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ==========================================
# 1. GROQ API CONFIGURATION
# ==========================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Add it in Render environment variables.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)


# ==========================================
# 2. PLANNER STATE
# ==========================================

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], add_messages]
    city: str
    interests: List[str]
    travel_style: str
    duration: str
    budget: str
    extra_requirements: str
    itinerary: str


# ==========================================
# 3. ITINERARY PROMPT
# ==========================================

itinerary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert AI travel planner.

        Create a personalized travel itinerary using the
        user's destination, interests, travel style,
        duration, budget and additional requirements.

        Include:
        1. Destination introduction.
        2. Day-by-day travel plan.
        3. Tourist attractions and activities.
        4. Food and restaurant suggestions.
        5. Transportation suggestions.
        6. Estimated budget considerations.
        7. Useful travel tips.

        Follow the requested travel style and budget.
        Use clear Markdown formatting.
        Do not invent confirmed bookings or real-time prices.
        """
    ),
    (
        "human",
        """
        Destination: {city}
        Interests: {interests}
        Travel Style: {travel_style}
        Trip Duration: {duration}
        Budget: {budget}
        Additional Requirements: {extra_requirements}

        Generate my personalized travel itinerary.
        """
    )
])


# ==========================================
# 4. LANGGRAPH NODES
# ==========================================

def input_city(state: PlannerState):
    if not state["city"].strip():
        raise ValueError("Please enter a destination city.")
    return {}


def input_interest(state: PlannerState):
    if not state["interests"]:
        raise ValueError("Please select at least one interest.")
    return {}


def input_itinerary(state: PlannerState):

    response = llm.invoke(
        itinerary_prompt.format_messages(
            city=state["city"],
            interests=", ".join(state["interests"]),
            travel_style=state["travel_style"],
            duration=state["duration"],
            budget=state["budget"],
            extra_requirements=state["extra_requirements"]
        )
    )

    itinerary = response.content

    return {
        "itinerary": itinerary,
        "messages": [AIMessage(content=itinerary)]
    }


# ==========================================
# 5. LANGGRAPH WORKFLOW
# ==========================================

workflow = StateGraph(PlannerState)

workflow.add_node("input_city", input_city)
workflow.add_node("input_interest", input_interest)
workflow.add_node("input_itinerary", input_itinerary)

workflow.set_entry_point("input_city")

workflow.add_edge("input_city", "input_interest")
workflow.add_edge("input_interest", "input_itinerary")
workflow.add_edge("input_itinerary", END)

graph = workflow.compile()


# ==========================================
# 6. GENERATE ITINERARY
# ==========================================

def generate_itinerary(
    city,
    interests,
    travel_style,
    duration,
    budget,
    extra_requirements
):

    if not city or not city.strip():
        return "Please enter a destination city."

    if not interests:
        return "Please select at least one interest."

    try:

        initial_state = {
            "messages": [
                HumanMessage(content=f"Plan a trip to {city}")
            ],
            "city": city,
            "interests": interests,
            "travel_style": travel_style,
            "duration": duration,
            "budget": budget,
            "extra_requirements": extra_requirements or "",
            "itinerary": ""
        }

        result = graph.invoke(initial_state)

        return result.get(
            "itinerary",
            "No itinerary was generated. Please try again."
        )

    except Exception as e:

        print("ERROR:", repr(e))

        return (
            "### Something went wrong\n"
            f"**Error:** `{type(e).__name__}: {str(e)}`"
        )


# ==========================================
# 7. CUSTOM CSS
# ==========================================

custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
}

#generate-btn {
    background: #176B5B;
    color: white;
    font-weight: bold;
}
"""


# ==========================================
# 8. GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="teal"
    ),
    css=custom_css,
    title="AI Travel Planner"
) as demo:

    gr.Markdown(
        """
        # ✈️ AI Travel Planner

        ### Your personal AI-powered travel companion

        Discover destinations, explore experiences, and
        generate personalized travel itineraries using Generative AI.
        """,
        elem_id="main-title"
    )

    gr.Markdown("---")

    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("### 🌍 Plan Your Journey")

            city = gr.Textbox(
                label="Destination City",
                placeholder="e.g. Paris, Tokyo, Goa"
            )

            interests = gr.CheckboxGroup(
                choices=[
                    "Adventure", "Food", "History", "Nature",
                    "Shopping", "Beaches", "Culture",
                    "Nightlife", "Photography"
                ],
                label="Your Interests"
            )

            travel_style = gr.Radio(
                choices=["Relaxed", "Balanced", "Fast-paced"],
                value="Balanced",
                label="Travel Style"
            )

            duration = gr.Dropdown(
                choices=[
                    "1-2 Days", "3-5 Days",
                    "6-7 Days", "1-2 Weeks"
                ],
                value="3-5 Days",
                label="Trip Duration"
            )

            budget = gr.Dropdown(
                choices=["Budget", "Moderate", "Luxury"],
                value="Moderate",
                label="Travel Budget"
            )

            extra_requirements = gr.Textbox(
                label="Additional Requirements",
                placeholder="e.g. Vegetarian food, family-friendly places...",
                lines=3
            )

            generate_btn = gr.Button(
                "✨ Generate My Itinerary",
                variant="primary",
                elem_id="generate-btn"
            )

        with gr.Column(scale=1):

            gr.Markdown("### 🗺️ Your Personalized Itinerary")

            output = gr.Markdown(
                value="Your travel itinerary will appear here."
            )

    gr.Markdown("---")

    gr.Markdown(
        """
        **Powered by:** Python · LangChain · LangGraph · Groq · Gradio

        *Intelligent travel planning powered by Generative AI.*
        """
    )

    generate_btn.click(
        fn=generate_itinerary,
        inputs=[
            city,
            interests,
            travel_style,
            duration,
            budget,
            extra_requirements
        ],
        outputs=output
    )


# ==========================================
# 9. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
