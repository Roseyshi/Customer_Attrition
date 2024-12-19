import os
import streamlit.web.bootstrap

if __name__ == "__main__":
    os.environ["STREAMLIT_SERVER_PORT"] = os.getenv("PORT", "8000")
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    streamlit.web.bootstrap.run("app.py", "", [], flag_options={})
