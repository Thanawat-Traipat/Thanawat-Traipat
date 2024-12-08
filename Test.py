import streamlit as st
import openai
import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import zipfile
from collections import Counter

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

def create_zip(key_points_df, quiz_df, pie_chart_img, histogram_img):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        csv_keypoints = key_points_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("1.keypoints.csv", csv_keypoints)

        csv_quiz = quiz_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("2.quiz.csv", csv_quiz)

        zip_file.writestr("4.piechart.png", pie_chart_img)
        zip_file.writestr("3.histogram.png", histogram_img)

    zip_buffer.seek(0)
    return zip_buffer.read()

def clean_key_phrases(key_phrases):
    phrase_counter = Counter({item['Phrase']: item['Frequency'] for item in key_phrases})
    return phrase_counter

st.markdown("""
# Private Tutor App 🎓

Welcome to the Private Tutor App – your personal study assistant powered by AI. This app helps you break down complex study material into manageable parts, with summaries, key points, visualizations, and quizzes to enhance your understanding.

### How does it work?
1. **Summarizes text**: Quickly condenses any text into a clear and concise summary.
2. **Extracts key points**: Pulls out the main ideas and explains them in simple terms.
3. **Visualizes data**: Generates frequency histograms of key phrases and pie charts to show key point distribution.
4. **Creates quizzes**: Builds 10 quiz questions to help you test your understanding.
5. **Multilingual Input, English Output**: You can input text in any language, and the app will process and provide output in English.
""")

st.markdown("## Input Your Study Material 📄")
st.markdown("Enter any text you'd like your private tutor to help you with, whether it's an article, study notes, or a textbook excerpt.")

user_input = st.text_area("Paste your text here for tutoring:", "Your text here", height=250)

detail_level = st.selectbox(
    "Choose the level of tutoring detail:",
    options=["Basic Overview", "Detailed Overview", "Thorough Analysis"],
    help="Select how much detail you'd like the tutor to focus on."
)

if detail_level == "Basic Overview":
    detail_instructions = "provide a very brief summary with only the key information."
    detail_instructions_2 = "provide the main key point and concise explanation."
elif detail_level == "Detailed Overview":
    detail_instructions = "provide a detailed summary with moderate depth and important concepts."
    detail_instructions_2 = "provide several key points with detailed explanations covering key concepts."
else:
    detail_instructions = "provide a thorough analysis with detailed explanations and insights."
    detail_instructions_2 = "provide an in-depth analysis with multiple key points and comprehensive explanations."

prompt = f"""
You are acting as a Private Tutor for the student. You will be given a text in any language, but you need to always respond in **English**. Complete the following tasks:

Step 1: Summarize the Text.
- {detail_instructions}

Step 2: Extract key points (provide key point and explanation).
- {detail_instructions_2}
Step 3: Extract key phrases and count their frequency of appearance in the text for a histogram.
- Return a list of key phrases with their corresponding frequency counts.
Step 4: Provide data for a pie chart based on key point importance.
- Return a list of key points with their corresponding percentage count (how much they cover in the paragraph and how important they are).
- Each key point is not equally important and represented.
Step 5: Generate 10 quiz questions.
- The questions should test the student’s knowledge from the text, to prepare them for the exam.
"""

if not user_api_key:
    st.sidebar.warning("Please provide your OpenAI API key to start using AI features.")

if not user_input:
    st.warning("Please provide some text for tutoring.")

if st.button('Get Tutoring') and user_input and client:
    with st.spinner("Your private tutor is preparing..."):
        ai_response = get_ai_response(prompt, user_input)

    try:
        response_data = json.loads(ai_response)

        # Fallback data for missing sections
        summary = response_data.get('Summary', "Summary not provided.")
        key_points = response_data.get('Key Points', [{"Key Point": "No Key Points", "Explanation": "No explanation provided."}])
        key_phrases = response_data.get('Key Phrases', [])
        pie_chart_data = response_data.get('Pie Chart Data', [{"Key Point": "No Key Points", "Percentage": 100}])
        quiz = response_data.get('Quiz', [{"Question": "What is this about?", "Answer": "Please refer to the summary.", "Explanation": "This is a general question."}])

        # Display summary
        st.markdown("## Summary 📜")
        st.write(summary)

        key_points_df = pd.DataFrame(key_points)
        key_points_df = key_points_df[['Key Point', 'Explanation']] 
        key_points_df.columns = ['Key Points', 'Explanation']
        key_points_df.index = key_points_df.index + 1

        st.markdown("## Key Points 🔑")
        st.markdown("""
        The key points break down the text into digestible parts. Each point includes a brief explanation to help you understand the material better.
        """)
        st.dataframe(key_points_df)

        quiz_df = pd.DataFrame(quiz)
        quiz_df.index = quiz_df.index + 1
        st.markdown("## Quiz Questions 📝")
        st.markdown("""
        To test your understanding, the AI has generated 10 quiz questions based on the key points of the text. This section helps reinforce your learning and ensures you are prepared for exams.
        """)
        st.dataframe(quiz_df)

        pie_labels = [point['Key Point'] for point in pie_chart_data]
        pie_sizes = [point['Percentage'] for point in pie_chart_data]
        fig, ax = plt.subplots()
        ax.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.axis('equal')

        pie_chart_img = io.BytesIO()
        plt.savefig(pie_chart_img, format='png')
        pie_chart_img.seek(0)

        # Create Key Phrase Frequency Histogram
        phrase_counter = clean_key_phrases(key_phrases)
        phrases, counts = zip(*phrase_counter.items())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(phrases, counts)
        ax.set_xlabel('Key Phrases')
        ax.set_ylabel('Frequency')
        ax.set_title('Frequency of Key Phrases')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(pad=3.0, bbox_inches='tight')

        histogram_img = io.BytesIO()
        plt.savefig(histogram_img, format='png')
        histogram_img.seek(0)

        # Data visualization tabs
        st.markdown("## Data Visualization 📊")
        st.markdown("""
        The histogram and pie chart give you a visual overview of the material, helping you see which key phrases appear most frequently and which key points are most emphasized.
        """)

        tab1, tab2 = st.tabs(["Key Phrase Frequency Histogram", "Key Point Pie Chart"])

        with tab1:
            st.image(histogram_img)

        with tab2:
            st.image(pie_chart_img)

        # Generate zip file for download
        zip_file = create_zip(key_points_df, quiz_df, pie_chart_img.getvalue(), histogram_img.getvalue())
        st.markdown("---")
        st.download_button(
            label="Download All Results (ZIP)",
            data=zip_file,
            file_name="PrivateTutorApp_output.zip",
            mime="application/zip"
        )

    except json.JSONDecodeError:
        st.error("Failed to parse the AI response into JSON. Please ensure the response follows the expected structure.")
