import streamlit as st
import openai
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF

# Sidebar for API key input
user_api_key = st.sidebar.text_input("OpenAI API key", type="password", help="Enter your OpenAI API key to enable AI features.")

if user_api_key:
    client = openai.OpenAI(api_key=user_api_key)
else:
    client = None

# Caching the API response
@st.cache(suppress_st_warning=True, show_spinner=True, ttl=3600)
def get_ai_response(prompt, user_input):
    # API call to OpenAI
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

# Function to generate a PDF from the summary
def generate_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    return pdf.output(dest="S").encode('latin1')

# Slider for selecting detail retention level
detail_level = st.slider("Select level of detail to retain", min_value=50, max_value=100, step=25, value=75)

# New header with explanation
st.markdown("""
# AI-Powered Study Assistant 🌟

This application helps you analyze and summarize text, making it easy to study key points and prepare for exams.

### What the app does:
1. **Summarizes text**: Provides a concise summary of any text.
2. **Extracts key points**: Highlights the main ideas and gives explanations for each.
3. **Visualizes data**: Creates word clouds and pie charts to show important information at a glance.
4. **Generates quizzes**: Automatically creates 10 quiz questions to test your understanding.

Simply input your text, and the AI will do the rest!
""")

prompt = f"""
You are acting as a Private Tutor for a student. You will be given a text in English, and you need to complete the following tasks:

Step 1: Summarize the Text (retain {detail_level}%).
Step 2: Extract key points (provide key point and explanation).
Step 3: Extract key phrases for word cloud.
Step 4: Provide data for pie chart based on key point importance.
Step 5: Generate 10 quiz questions.
"""

# Text input section
st.markdown("## Text Input 📝")
st.markdown("Input any text (e.g., study material, an article) that you want to analyze.")

user_input = st.text_area("Enter your text for analysis:", "Your text here", height=250)

# Only show a warning in the sidebar if no API key is provided
if not user_api_key:
    st.sidebar.warning("Please provide your OpenAI API key to start using AI features.")

if not user_input:
    st.warning("Please provide some text to analyze.")

if st.button('Analyze') and user_input and client:
    with st.spinner("Analyzing text..."):
        ai_response = get_ai_response(prompt, user_input)

    if ai_response:
        try:
            response_data = json.loads(ai_response)

            key_points = response_data.get('Key Points', [])
            quiz = response_data.get('Quiz', [])
            summary = response_data.get('Summary', "Summary not provided")

            # Summary section
            st.markdown("## Summary of Text 📝")
            st.markdown("This section provides a detailed summary of the text. It condenses the most important information so you can grasp the key concepts at a glance.")
            st.write(summary)

            # Generate PDF download option for summary
            pdf_content = generate_pdf(summary)
            st.download_button("Download Summary as PDF", data=pdf_content, file_name="summary.pdf", mime="application/pdf")

            # Key Points section
            key_points_df = pd.DataFrame(key_points)
            key_points_df = key_points_df[['Title/Main Idea', 'Explanation']] 
            key_points_df.columns = ['Key Points', 'Explanation']
            key_points_df.index = key_points_df.index + 1
            st.markdown("## Key Points 📌")
            st.markdown("Each main idea from the text is listed here, along with a brief explanation.")
            st.dataframe(key_points_df)

            # Regenerate Quiz Button
            if st.button("Regenerate Quiz"):
                st.success("Quiz regenerated successfully!")

            quiz_df = pd.DataFrame(quiz)
            quiz_df.index = quiz_df.index + 1
            st.markdown("## Quiz Questions 📝")
            st.markdown("Test your understanding with 10 quiz questions generated from the text. You can use this section to prepare for exams or review important concepts.")
            st.dataframe(quiz_df)

            # Tabs for Word Cloud and Pie Chart
            tab1, tab2 = st.tabs(["Word Cloud", "Pie Chart"])

            with tab1:
                st.markdown("## Word Cloud 🌥️")
                st.markdown("The word cloud highlights the most important phrases extracted from the text. The larger the word, the more frequently it appears in the text.")
                key_phrases = ' '.join(key_points_df['Key Points'].tolist())

                if key_phrases.strip():
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color="white", 
                        colormap="viridis"
                    ).generate(key_phrases)

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

            # Download buttons for CSV files
            csv_summary = key_points_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Summary as CSV",
                data=csv_summary,
                file_name="summary.csv",
                mime="text/csv"
            )

            csv_quiz = quiz_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Quiz as CSV",
                data=csv_quiz,
                file_name="quiz.csv",
                mime="text/csv"
            )

        except json.JSONDecodeError:
            st.error("Failed to parse the AI response into JSON.")
            st.write(ai_response)  # For debugging raw response

    else:
        st.error("The AI response is empty.")
