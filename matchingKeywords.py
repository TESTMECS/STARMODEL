import re

def analyze_star_response(response):
    # Define keywords for each STAR component
    star_components = {
        "Situation": ["situation", "background", "context", "challenge"],
        "Task": ["task", "goal", "objective", "responsibility"],
        "Action": ["action", "step", "approach", "method", "organized"],
        "Result": ["result", "outcome", "impact", "achievement", "success"]
    }
    
    # Split the response into sentences
    sentences = re.split(r'\.|\?|!', response)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Analyze sentences for STAR components
    analysis = {key: [] for key in star_components}
    for sentence in sentences:
        for component, keywords in star_components.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                analysis[component].append(sentence)
                break
    
    # Output results
    result_summary = {}
    for component, matched_sentences in analysis.items():
        if matched_sentences:
            result_summary[component] = matched_sentences
        else:
            result_summary[component] = ["No clear statement found."]
    
    return result_summary


# Example Usage
response = """
In my previous role, I was faced with a challenging situation where our team’s project deadline was suddenly moved up by two weeks. 
My task was to coordinate with all departments to re-prioritize our goals without compromising on quality. 
I organized daily stand-up meetings and implemented a streamlined workflow to enhance productivity. 
As a result, we delivered the project on time, receiving praise from the client for our efficiency.
"""

analysis_result = analyze_star_response(response)

# Display the results
for component, sentences in analysis_result.items():
    print(f"\n{component}:")
    for sentence in sentences:
        print(f"  - {sentence}")
