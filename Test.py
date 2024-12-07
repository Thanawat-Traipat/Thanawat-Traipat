import streamlit as st
import openai
import pandas as pd
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

if not user_api_key:
    st.warning("Please provide your OpenAI API key.")
else:
    client = openai.OpenAI(api_key=user_api_key)

    prompt = """Act as a Private Tutor for the student. You will be given a text,
                and you need to complete the following tasks in a structured, step-by-step manner:

                Step 1: Summarize the Text
                - Provide a brief summary of the entire text.
                - Ensure the summary is concise but retains key concepts.
                
                Step 2: Identify Key Points
                - For each key point, provide the following:
                    1. Title/Main Idea
                    2. Key Concepts
                    3. Short Summary of the point

                Step 3: Extract Key Phrases
                - Extract key phrases to create a word cloud.

                Step 4: Generate Pie Chart Data
                - Organize data to show the importance of each point in a pie chart.

                Step 5: Create Quiz Questions
                - Create 10 quiz questions based on the Key Concepts.

                The response should be a JSON object with the following structure:
                {
                    "Summary": "brief summary of the text",
                    "Key Points": [
                        {
                            "Title/Main Idea": "title",
                            "Key Concepts": ["concept1", "concept2", ...],
                            "Explanation": "short summary of the key point"
                        },
                        ...
                    ],
                    "Quiz": [
                        {
                            "Question": "question",
                            "Answer": "correct answer",
                            "Answer Explanation": "explanation of the answer"
                        },
                        ...
                    ]
                }

                **Please ensure that even if there are no key points or quiz questions, you return the keys 'Key Points' and 'Quiz' with empty arrays.**
                """

    st.title("Summarization and Exam Preparation Tutor")
    st.markdown("<h2 style='text-align: center;'>AI-Powered Study Tool</h2>", unsafe_allow_html=True)

    user_input = st.text_area("Enter your text for analysis:", "Your text here", height=250)

    if st.button('Analyze'):
        if not user_input:
            st.error("Please enter some text to analyze.")
        else:
            messages_so_far = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages_so_far
                )
                
                ai_response = response.choices[0].message.content

                if ai_response:
                    try:
                        # Attempt to parse the response as JSON
                        response_data = json.loads(ai_response)
                        
                        # Ensure 'Key Points' and 'Quiz' exist and are not missing
                        key_points = response_data.get('Key Points', [])
                        quiz = response_data.get('Quiz', [])
                        summary = response_data.get('Summary', "No summary provided.")

                        # If 'Key Points' or 'Quiz' are empty, we provide fallback empty values
                        if not key_points:
                            key_points = [{"Title/Main Idea": "No Key Points", "Key Concepts": [], "Explanation": "No explanation"}]
                        if not quiz:
                            quiz = [{"Question": "No Questions", "Answer": "N/A", "Answer Explanation": "No explanation"}]

                        # Process Key Points into DataFrame
                        key_points_df = pd.DataFrame(key_points)

                        key_points_df = key_points_df[['Title/Main Idea', 'Explanation']] 
                        key_points_df.columns = ['Key Points', 'Explanation']  

                        # Display Summary
                        st.subheader("Summary of Text")
                        st.write(summary)

                        # Display Key Points and Explanation
                        st.subheader("Key Points")
                        st.dataframe(key_points_df)

                        # Display Quiz
                        quiz_df = pd.DataFrame(quiz)
                        st.subheader("Quiz Questions")
                        st.dataframe(quiz_df)

                        # Generate Word Cloud for Key Phrases
                        st.subheader("Word Cloud for Key Phrases")
                        key_phrases = ' '.join(key_points_df['Key Points'].tolist())
                        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(key_phrases)
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        st.pyplot(plt.gcf())

                        # Generate Pie Chart for Key Points
                        st.subheader("Pie Chart of Key Points Representation")
                        pie_labels = key_points_df['Key Points']
                        pie_sizes = [len(k) for k in key_points_df['Explanation']]  
                        fig, ax = plt.subplots()
                        ax.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                        ax.axis('equal')
                        st.pyplot(fig)

                        # Download Options
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
                        st.write(ai_response)  # Print raw response for debugging
                else:
                    st.error("The AI response is empty.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
