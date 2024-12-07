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
                """

    st.title('Summarization and Exam Preparation Tutor')
    st.markdown('Input the text that you want to analyze. \n\
                The AI will generate a summary, extract key phrases, create a word cloud, and generate quiz questions.')

    user_input = st.text_area("Enter text for analysis:", "Your text here")

    if st.button('Submit'):
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
                
                st.markdown('**AI response:**')
                ai_response = response.choices[0].message.content
                
                response_data = json.loads(ai_response)

                key_points_df = pd.DataFrame(response_data['Key Points'])

                key_points_df = key_points_df[['Title/Main Idea', 'Summary']] 
                key_points_df.columns = ['Key Points', 'Explanation']  

                visualization_choice = st.radio("Choose Visualization:", ("Word Cloud", "Pie Chart"))

                if visualization_choice == "Word Cloud":
                    st.subheader("Word Cloud for Key Phrases")
                    key_phrases = ' '.join(key_points_df['Key Points'].tolist())
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(key_phrases)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt.gcf())

                elif visualization_choice == "Pie Chart":
                    st.subheader("Pie Chart of Key Points Representation")
                    pie_labels = key_points_df['Key Points']
                    pie_sizes = [len(k) for k in key_points_df['Explanation']]
                    fig, ax = plt.subplots()
                    ax.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                    ax.axis('equal')
                    st.pyplot(fig)

                content_choice = st.radio("Choose Content:", ("Summary", "Quiz"))

                if content_choice == "Summary":
                    st.subheader("Summary of Text")
                    st.dataframe(key_points_df)

                    csv_summary = key_points_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Summary as CSV",
                        data=csv_summary,
                        file_name="summary.csv",
                        mime="text/csv"
                    )

                elif content_choice == "Quiz":
                    st.subheader("Quiz Questions")
                    quiz_df = pd.DataFrame(response_data['Quiz'])
                    st.dataframe(quiz_df)

                    csv_quiz = quiz_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Quiz as CSV",
                        data=csv_quiz,
                        file_name="quiz.csv",
                        mime="text/csv"
                    )

                    if st.checkbox("Show Answer Keys"):
                        quiz_df["Answer Key"] = quiz_df["Answer Explanation"]
                        st.dataframe(quiz_df)

                        csv_quiz_with_answers = quiz_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Quiz with Answer Keys as CSV",
                            data=csv_quiz_with_answers,
                            file_name="quiz_with_answers.csv",
                            mime="text/csv"
                        )
                        
            except json.JSONDecodeError:
                st.error("The response from the AI couldn't be parsed into a table format.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
