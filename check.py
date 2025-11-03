import google.generativeai as genai
genai.configure(api_key="AIzaSyAgFhu7_ZSUS0fFt3MxugteJj2NsoLQPwM")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
