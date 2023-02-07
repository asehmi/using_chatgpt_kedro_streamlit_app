# Using ChatGPT to build a Kedro ML pipeline and Streamlit frontend

I recently came across an open-source Python DevOps framework Kedro and thought, “Why not  have ChatGPT teach me how to use it to build some ML/DevOps automation?” The idea was to:
1. Ask ChatGPT some basic questions about Kedro.
2. Ask it to use more advanced features in the Kedro framework.
3. Write my questions with hints and phrases that encouraged explanations of advanced Kedro features (to evolve incrementally as if I were taught by a teacher).

Kedro has some pipeline visualization capabilities, so I wondered:
- Could ChatGPT show me how to display pipeline graphs in Streamlit?
- Could ChatGPT build me an example ML model and explicitly refer to it in the Kedro pipeline?
- What does it take to scale the pipeline, and perform pipeline logging, monitoring, and error handling?
- Could I connect Kedro logs to a cloud-based logging service?
- Could ChatGPT contrast Kedro with similar (competing) products and services and show me how the pipeline it developed earlier could be implemented in one of them?

I wrote a blog post with annotated responses to the answers I got to my questions. I was super impressed and decided to implement the Kedro pipeline and Streamlit application as planned from what I learned. This repository contains all the code for the application. 
