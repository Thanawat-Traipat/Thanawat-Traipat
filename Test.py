import streamlit as st
import openai
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import zipfile

# Sidebar input for OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password", help="Enter your OpenAI API key to enable AI features.")

if user_api_key:
    client = openai.OpenAI(api_key=user_api_key)
else:
    client = None

# Function to call OpenAI API
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

# Function to create a zip file
def create_zip(key_points_df, quiz_df, pie_chart_img, wordcloud_img):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        # Add Key Points CSV
        csv_keypoints = key_points_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("1.keypoints.csv", csv_keypoints)

        # Add Quiz CSV
        csv_quiz = quiz_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("2.quiz.csv", csv_quiz)

        # Add Pie Chart Image
        zip_file.writestr("4.piechart.png", pie_chart_img)

        # Add Word Cloud Image
        zip_file.writestr("3.wordcloud.png", wordcloud_img)

    zip_buffer.seek(0)
    return zip_buffer.read()

# Displaying Header and Instructions
st.markdown("""
# AI-Powered Study Assistant 🌟

This application helps you analyze and summarize text, making it easy to study key points and prepare for exams.

### What the app does:
1. **Summarizes text**: Provides a concise summary of any text, no matter the language.
2. **Extracts key points**: Highlights the main ideas and gives explanations for each, regardless of the input language.
3. **Visualizes data**: Creates word clouds and pie charts to show important information at a glance.
4. **Generates quizzes**: Automatically creates 10 quiz questions to test your understanding.
5. **Multilingual Input, English Output**: Input any text in any language (e.g., Thai, Japanese, Spanish, etc.), and the app will provide the output **in English**.

Simply input your text, and the AI will do the rest!
""")

# Text input area
st.markdown("## Text Input 📝")
st.markdown("Input any text (e.g., study material, an article) that you want to analyze.")
user_input = st.text_area("Enter your text for analysis:", "Your text here", height=250)

# Option to select level of detail
detail_level = st.selectbox(
    "Choose the level of analysis detail:",
    options=["Basic Overview", "Detailed Overview", "Thorough Analysis"],
    help="Select how much detail you want in the analysis."
)

# AI prompt that changes based on the selected detail level
if detail_level == "Basic Overview":
    detail_instructions = "provide a very brief summary with only the key information."
elif detail_level == "Detailed Overview":
    detail_instructions = "provide a detailed summary with moderate depth and important concepts."
else:
    detail_instructions = "provide a thorough analysis with detailed explanations and insights."

prompt = f"""
You are acting as a Private Tutor for the student. You will be given a text in any language, but you need to always respond in **English**. Complete the following tasks:

Step 1: Summarize the Text ({detail_level}).
- {detail_instructions}

Step 2: Extract key points (provide key point and explanation).
Step 3: Extract key phrases for word cloud.
Step 4: Provide data for pie chart based on key point importance.
Step 5: Generate 10 quiz questions.

### **Important Instructions**:
1. **Complete all sections**: Every part of the response (summary, key points, key phrases, pie chart data, quiz questions) must be included. If a section cannot be generated, provide an empty object or an empty list to ensure the section is not omitted.
2. **Formatting**: The output must always adhere to the structured JSON format shown below. Use empty placeholders (like `{{}}` for objects or `[]` for lists) where necessary.

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
"""

if not user_api_key:
    st.sidebar.warning("Please provide your OpenAI API key to start using AI features.")

if not user_input:
    st.warning("Please provide some text to analyze.")

# Analyze button and output processing
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

        # Display Summary with Explanation
        st.markdown("## Summary of Text 📝")
        st.markdown("""
        The summary provides a brief, high-level overview of the text. Depending on the selected level of detail, the summary will include key points or provide more thorough explanations.
        """)
        st.write(summary)

        # Key Points DataFrame with Explanation
        key_points_df = pd.DataFrame(key_points)
        key_points_df = key_points_df[['Key Point', 'Explanation']] 
        key_points_df.columns = ['Key Points', 'Explanation']
        key_points_df.index = key_points_df.index + 1

        st.markdown("## Key Points 📌")
        st.markdown("""
        The key points highlight the most important ideas from the text, along with brief explanations for each point. This helps you focus on the essential concepts for studying.
        """)
        st.dataframe(key_points_df)

        # Quiz DataFrame with Explanation
        quiz_df = pd.DataFrame(quiz)
        quiz_df.index = quiz_df.index + 1
        st.markdown("## Quiz Questions 📝")
        st.markdown("""
        To test your understanding, the AI has generated 10 quiz questions based on the key points of the text. This section helps reinforce your learning and ensures you are prepared for exams.
        """)
        st.dataframe(quiz_df)

        # Create the Pie Chart
        pie_labels = [point['Key Point'] for point in pie_chart_data]
        pie_sizes = [point['Percentage'] for point in pie_chart_data]
        fig, ax = plt.subplots()
        ax.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.axis('equal')

        pie_chart_img = io.BytesIO()
        plt.savefig(pie_chart_img, format='png')
        pie_chart_img.seek(0)

        # Create the WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(key_phrases))
        wordcloud_img = io.BytesIO()
        wordcloud.to_image().save(wordcloud_img, format='PNG')
        wordcloud_img.seek(0)

        # Visualization Tabs with Explanation
        tab1, tab2 = st.tabs(["Word Cloud", "Pie Chart"])

        with tab1:
            st.markdown("## Word Cloud 🌥️")
            st.markdown("""
            The word cloud shows the most important words and phrases extracted from the text. The bigger the word, the more frequently it appears in the text. This gives you a quick visual of key themes.
            """)
            key_phrases_text = ' '.join(key_phrases)

            if key_phrases_text.strip():
                wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(key_phrases_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt.gcf())
            else:
                st.warning("No key phrases were extracted for the word cloud.")

        with tab2:
            st.markdown("## Pie Chart 📊")
            st.markdown("""
            The pie chart breaks down the relative importance of each key point based on how frequently it is mentioned or emphasized in the text.
            """)
            if not key_points_df.empty:
                pie_labels = key_points_df['Key Points']
                pie_sizes = [len(k) for k in key_points_df['Explanation']]
                fig, ax = plt.subplots()
                ax.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

        # Move Download All Button to the Bottom
        zip_file = create_zip(key_points_df, quiz_df, pie_chart_img.getvalue(), wordcloud_img.getvalue())
        st.markdown("---")
        st.download_button(
            label="Download All (ZIP)",
            data=zip_file,
            file_name="PrivateTutorApp_output.zip",
            mime="application/zip"
        )

    except json.JSONDecodeError:
        st.error("Failed to parse the AI response into JSON. Please ensure the response follows the expected structure.")
