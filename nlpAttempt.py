import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_trf")

def analyze_star_response_nlp(response):
    # Define STAR prompts for semantic analysis
    prompts = {
        "Situation": ["Describe the context or background.", "What was the challenge?"],
        "Task": ["What was your role or objective?", "What were you responsible for?"],
        "Action": ["What steps did you take?", "How did you approach the situation?"],
        "Result": ["What was the outcome?", "What impact did it have?"]
    }
    
    # Process the response with SpaCy
    doc = nlp(response)
    analysis = {key: [] for key in prompts.keys()}

    for sent in doc.sents:
        for component, questions in prompts.items():
            for question in questions:
                # Check semantic similarity between sentence and questions
                question_doc = nlp(question)
                similarity = sent.similarity(question_doc)
                if similarity > 0.75:  # Threshold for similarity
                    analysis[component].append(sent.text)
                    break

    # Final analysis with missing checks
    result_summary = {}
    for component, sentences in analysis.items():
        result_summary[component] = sentences if sentences else ["No clear statement found."]
    
    return result_summary


# Example response
response = """
In my previous role, our team faced a sudden challenge where the project deadline was moved up by two weeks. 
I was responsible for coordinating with all departments to adjust our goals without compromising quality. 
To address this, I organized daily stand-ups and created a workflow to streamline processes. 
This resulted in completing the project on time and receiving client praise for efficiency.
"""

# Analyze the response
result = analyze_star_response_nlp(response)


# Display the results
for component, sentences in result.items():
    print(f"\n{component}:")
    for sentence in sentences:
        print(f"  - {sentence}")
