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
                
                # Rest of the prompt here...
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
                    response_data = json.loads(ai_response)
                    
                    # Displaying Summary
                    st.subheader("Summary of Text")
                    st.write(response_data["Summary"])

                    # Key Points Table
                    key_points_df = pd.DataFrame(response_data['Key Points'])
                    key_points_df = key_points_df[['Title/Main Idea', 'Explanation']]
                    key_points_df.columns = ['Key Points', 'Explanation']
                    st.subheader("Key Points and Explanations")
                    st.dataframe(key_points_df, use_container_width=True)

                    # Quiz Questions
                    quiz_df = pd.DataFrame(response_data['Quiz'])
                    st.subheader("Quiz Questions")
                    st.dataframe(quiz_df, use_container_width=True)

                    # Generate Word Cloud
                    st.subheader("Word Cloud for Key Phrases")
                    key_phrases = ' '.join(key_points_df['Key Points'].tolist())
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(key_phrases)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt.gcf())

                    # Pie Chart Visualization
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

            except Exception as e:
                st.error(f"An error occurred: {e}")
