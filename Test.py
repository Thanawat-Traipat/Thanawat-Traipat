import streamlit as st
import openai
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF

user_api_key = st.sidebar.text_input("OpenAI API key", type="password", help="Enter your OpenAI API key to enable AI features.")

if user_api_key:
    client = openai.OpenAI(api_key=user_api_key)
else:
    client = None

def get_ai_response(prompt, user_input):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

def generate_pdf(content, title):
    pdf = FPDF()
    pdf.add_page()
    
    # Set a basic font (Arial is a default Latin font supported by FPDF)
    pdf.set_font("Arial", size=12)  # No need for custom Unicode fonts
    
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)  # Line break

    # Add the content to the PDF (English text)
    for line in content.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)

    return pdf.output(dest="S").encode('latin1')  # Default encoding for PDF download

# Text input area
st.markdown("## Text Input 📝")
st.markdown("Input any text (e.g., study material, an article) that you want to analyze.")
user_input = st.text_area("Enter your text for analysis:", "Your text here", height=250)

# Slider to adjust detail level
detail_level = st.slider(
    "Select level of detail to retain",
    min_value=40,
    max_value=100,
    step=10,
    value=60,
    help="Select how much detail you want to retain in the summary and quiz."
)

# AI prompt that changes based on slider value
prompt = f"""
You are acting as a Private Tutor for the student. You will be given a text in any language, but you need to always respond in **English**. Complete the following tasks:

Step 1: Summarize the Text (retain {detail_level}%).
- If the detail level is high (e.g., > 70%), provide a detailed summary with thorough explanations and insights.
- If the detail level is low (e.g., <= 70%), provide a concise summary with only the key information.

Step 2: Extract key points (provide key point and explanation).
Step 3: Extract key phrases for word cloud.
Step 4: Provide data for pie chart based on key point importance.
Step 5: Generate 10 quiz questions.

### **Important Instructions**:
1. **Complete all sections**: Every part of the response (summary, key points, key phrases, pie chart data, quiz questions) must be included. If a section cannot be generated, provide an empty object or an empty list to ensure the section is not omitted.
2. **Formatting**: The output must always adhere to the structured JSON format shown below. Use empty placeholders (like `{{}}` for objects or `[]` for lists) where necessary.
3. **If in doubt**: When unsure, provide **empty JSON structures** like `{{}}` for missing sections.

Output Format:

{{
    "Summary": "brief summary of the text",
    "Key Points": [
        {{
            "Key Point": "main idea or title",
            "Explanation": "concise explanation"
        }},
        ...
    ],
    "Key Phrases": [],
    "Pie Chart Data": [
        {{
            "Key Point": "key point title",
            "Percentage": percentage_value
        }},
        ...
    ],
    "Quiz": [
        {{
            "Question": "quiz question",
            "Answer": "correct answer",
            "Explanation": "answer explanation"
        }},
        ...
    ]
}}

If any section is missing, always include empty objects or empty lists to ensure proper formatting.
"""

if not user_api_key:
    st.sidebar.warning("Please provide your OpenAI API key to start using AI features.")

if not user_input:
    st.warning("Please provide some text to analyze.")

# Store results in session state
if st.button('Analyze') and user_input and client:
    with st.spinner("Analyzing text..."):
        ai_response = get_ai_response(prompt, user_input)

    try:
        # Ensure the AI response is valid JSON
        response_data = json.loads(ai_response)

        # Fallback data for missing sections
        summary = response_data.get('Summary', "Summary not provided.")
        key_points = response_data.get('Key Points', [{"Key Point": "No Key Points", "Explanation": "No explanation provided."}])
        key_phrases = response_data.get('Key Phrases', [])
        pie_chart_data = response_data.get('Pie Chart Data', [{"Key Point": "No Key Points", "Percentage": 100}])
        quiz = response_data.get('Quiz', [{"Question": "What is this about?", "Answer": "Please refer to the summary.", "Explanation": "This is a general question."}])

        # Save the data in session_state
        st.session_state.summary = summary
        st.session_state.key_points = key_points
        st.session_state.key_phrases = key_phrases
        st.session_state.pie_chart_data = pie_chart_data
        st.session_state.quiz = quiz

        # Display the Summary
        st.markdown("## Summary of Text 📝")
        st.markdown("This section provides a detailed summary of the text. It condenses the most important information so you can grasp the key concepts at a glance.")
        st.write(st.session_state.summary)

        pdf_content = generate_pdf(st.session_state.summary, "Summary")
        st.download_button("Download Summary as PDF", data=pdf_content, file_name="summary.pdf", mime="application/pdf")

        # Display Key Points
        key_points_df = pd.DataFrame(st.session_state.key_points)
        key_points_df = key_points_df[['Key Point', 'Explanation']] 
        key_points_df.columns = ['Key Points', 'Explanation']
        key_points_df.index = key_points_df.index + 1

        st.markdown("## Key Points 📌")
        st.markdown("Each main idea from the text is listed here, along with a brief explanation.")
        st.dataframe(key_points_df)

        # Key Points CSV Download
        csv_summary = key_points_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download full table (CSV)",
            data=csv_summary,
            file_name="summary.csv",
            mime="text/csv"
        )

        # Key Points PDF Download
        key_points_pdf_content = generate_pdf(key_points_df.to_string(), "Key Points and Explanations")
        st.download_button("Download Key Points as PDF", data=key_points_pdf_content, file_name="key_points.pdf", mime="application/pdf")

        # Display Quiz
        quiz_df = pd.DataFrame(st.session_state.quiz)
        quiz_df.index = quiz_df.index + 1
        st.markdown("## Quiz Questions 📝")
        st.markdown("Test your understanding with 10 quiz questions generated from the text. You can use this section to prepare for exams or review important concepts.")
        st.dataframe(quiz_df)

        # Quiz CSV Download
        csv_quiz = quiz_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download full table (CSV)",
            data=csv_quiz,
            file_name="quiz.csv",
            mime="text/csv"
        )

        pdf_quiz_content = generate_pdf(quiz_df.to_string(), "Quiz Questions")
        st.download_button("Download Quiz as PDF", data=pdf_quiz_content, file_name="quiz.pdf", mime="application/pdf")

        # Visualization Tabs
        tab1, tab2 = st.tabs(["Word Cloud", "Pie Chart"])

        with tab1:
            st.markdown("## Word Cloud 🌥️")
            st.markdown("The word cloud highlights the most important phrases extracted from the text. The larger the word, the more frequently it appears in the text.")
            key_phrases_text = ' '.join(st.session_state.key_phrases)

            if key_phrases_text.strip():
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color="white", 
                    colormap="viridis"
                ).generate(key_phrases_text)

                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt.gcf())
            else:
                st.warning("No key phrases were extracted for the word cloud.")

        with tab2:
            st.markdown("## Pie Chart 📊")
            st.markdown("The pie chart visually represents the relative importance of each key point based on its presence and significance in the text.")
            if not key_points_df.empty:
                pie_labels = key_points_df['Key Points']
                pie_sizes = [len(k) for k in key_points_df['Explanation']]
                if sum(pie_sizes) > 0:
                    fig, ax = plt.subplots()
                    ax.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                    ax.axis('equal')
                    st.pyplot(fig)
                else:
                    st.warning("No valid data available to generate the pie chart.")
            else:
                st.warning("No key points available to generate the pie chart.")

    except json.JSONDecodeError:
        st.error("Failed to parse the AI response into JSON. Please ensure the response follows the expected structure.")
