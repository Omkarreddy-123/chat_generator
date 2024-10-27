import streamlit as st
import pandas as pd
from typing import List, Dict, Set
import random
import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

class ChatAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("Please install the spacy model using: python -m spacy download en_core_web_sm")
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer()
        
        self.custom_stop_words = {
            'the', 'what', 'is', 'a', 'there', 'has', 'here', 'this', 'that',
            'these', 'those', 'am', 'be', 'been', 'being', 'was', 'were',
            'will', 'would', 'should', 'can', 'could', 'may', 'might',
            'must', 'shall', 'to', 'of', 'in', 'for', 'on', 'with'
        }
        
        for word in self.custom_stop_words:
            self.nlp.vocab[word].is_stop = True
        
        self.complexity_weights = {
            'basic': 0.5,
            'intermediate': 0.3,
            'advanced': 0.2
        }
        
        self.question_templates = {
            'basic': [
                "What is the definition of {concept}?",
                "Which of the following best describes {concept}?",
                "What role does {concept} play in {context}?",
                "What is the function of {concept} in {context}?",
                "How would you explain {concept}?"
            ],
            'intermediate': [
                "How does {concept} differ from {related_concept}?",
                "What are the benefits of using {concept} in {context}?",
                "In what situations would you use {concept} over {related_concept}?",
                "What are the key features of {concept} in {context}?",
                "How is {concept} applied in {context}?"
            ],
            'advanced': [
                "What are the limitations of {concept} in {context}?",
                "How does {concept} address issues in {related_concept}?",
                "What is the impact of {concept} in solving problems in {context}?",
                "What challenges exist when applying {concept} in {context}?",
                "How does {concept} compare to {related_concept} in {context}?"
            ]
        }
        
        self.concept_definitions = defaultdict(list)

    def preprocess(self, text: str) -> str:
        """
        Preprocess text by removing stop words and punctuation, and lemmatizing tokens.
        """
        filtered = []
        for token in self.nlp(text):
            if token.is_stop or token.is_punct:
                continue
            filtered.append(token.lemma_)
        return " ".join(filtered)

    def _extract_semantic_info(self, messages: List[str]) -> List[Dict]:
        """
        Extracts key terms from chat messages using preprocessing and TF-IDF.
        """
        # Preprocess all messages
        filtered_messages = [self.preprocess(msg) for msg in messages]
        
        # Apply TF-IDF vectorization
        if not filtered_messages:
            return []
            
        vectorized_messages = self.tfidf.fit_transform(filtered_messages)
        
        # Get feature names and their scores
        feature_names = self.tfidf.get_feature_names_out()
        tfidf_scores = vectorized_messages.toarray()
        
        semantic_info = []
        for idx, (original_msg, filtered_msg) in enumerate(zip(messages, filtered_messages)):
            doc = self.nlp(original_msg)
            
            # Extract concepts with their TF-IDF scores
            concepts = []
            for chunk in doc.noun_chunks:
                if self._is_valid_concept(chunk):
                    cleaned_term = self.preprocess(chunk.text)
                    if cleaned_term:
                        # Get TF-IDF score for the term if it exists in features
                        importance = 0.0
                        for word in cleaned_term.split():
                            if word in feature_names:
                                word_idx = list(feature_names).index(word)
                                importance += tfidf_scores[idx][word_idx]
                        
                        if importance > 0:  # Only add terms with non-zero TF-IDF scores
                            concepts.append({
                                "term": cleaned_term,
                                "context": self._get_context(doc, chunk.root),
                                "sentence": original_msg,
                                "importance": importance
                            })
            
            # Sort concepts by TF-IDF importance
            concepts.sort(key=lambda x: x['importance'], reverse=True)
            
            if concepts:  # Only add messages that have valid concepts
                info = {'text': original_msg, 'concepts': concepts}
                semantic_info.append(info)
                self._build_concept_definitions(info)
        
        return semantic_info

    def _is_valid_concept(self, chunk: spacy.tokens.Span) -> bool:
        """Check if a noun chunk is a valid concept."""
        preprocessed = self.preprocess(chunk.text)
        return bool(preprocessed)

    def _get_context(self, doc, token, window=5):
        """Extract context using preprocessing."""
        start = max(token.i - window, 0)
        end = min(token.i + window + 1, len(doc))
        context_text = doc[start:end].text
        return self.preprocess(context_text)

    def _build_concept_definitions(self, info):
        """Builds concept definitions for later use."""
        for concept in info['concepts']:
            term = concept['term']
            if term:  # Only add if term is not empty after preprocessing
                self.concept_definitions[term].append({
                    'definition': concept['context'],
                    'sentence': concept['sentence']
                })

    def _generate_mcq(self, concept_info: Dict, complexity: str = 'basic') -> Dict:
        """Generates an MCQ using preprocessed concepts."""
        concept = concept_info['term']
        context = concept_info['context']
        
        # Get related concepts from the concept definitions
        related_concepts = [key for key in self.concept_definitions.keys() if key != concept]
        related_concept = random.choice(related_concepts) if related_concepts else "alternative concept"
        
        # Generate question text using template
        question_text = random.choice(self.question_templates[complexity]).format(
            concept=concept, 
            context=context, 
            related_concept=related_concept
        )
        
        # Generate correct answer
        correct_answer = f"{concept}: {self._get_definition_from_context(concept)}"
        
        # Generate incorrect options
        incorrect_options = self._generate_incorrect_options(correct_answer, concept, related_concept)
        
        # Combine and shuffle options
        options = [correct_answer] + incorrect_options
        random.shuffle(options)
        
        return {
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "type": "mcq",
            "complexity": complexity,
            "concept": concept
        }

    def _generate_incorrect_options(self, correct_answer: str, concept: str, related_concept: str, num_options: int = 3) -> List[str]:
        """Generate plausible but incorrect options for the MCQ."""
        incorrect_phrases = [
            lambda c, rc: f"{c}: is similar to {rc} but serves a different purpose",
            lambda c, rc: f"{c}: represents a subset of {rc}",
            lambda c, rc: f"{c}: provides an alternative approach to {rc}",
            lambda c, rc: f"{c}: contrasts with {rc} in fundamental ways",
            lambda c, rc: f"{c}: combines elements from both {rc} and other concepts"
        ]
        
        incorrect_options = set()
        attempts = 0
        max_attempts = 10  # Prevent infinite loops
        
        while len(incorrect_options) < num_options and attempts < max_attempts:
            phrase_func = random.choice(incorrect_phrases)
            option = phrase_func(concept, related_concept)
            
            if option != correct_answer and option not in incorrect_options:
                incorrect_options.add(option)
            
            attempts += 1
        
        # If we couldn't generate enough unique options, fill with generic ones
        while len(incorrect_options) < num_options:
            generic_option = f"{concept}: serves a different purpose than described"
            if generic_option not in incorrect_options:
                incorrect_options.add(generic_option)
        
        return list(incorrect_options)

    def _get_definition_from_context(self, concept: str) -> str:
        """Get the most comprehensive definition for a concept from its contexts."""
        definitions = self.concept_definitions.get(concept, [])
        if definitions:
            # Choose the longest definition as it's likely to be most informative
            best_def = max(definitions, key=lambda x: len(x['definition']))
            return best_def['definition'].strip() + '.'
        return f"represents a key concept in the given context"

    def generate_mcqs(self, messages: List[str], num_questions: int = 5) -> List[Dict]:
        """
        Generate multiple choice questions from the given messages.
        
        Args:
            messages: List of text messages to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of dictionaries containing question information
        """
        # Extract semantic information from messages
        semantic_info = self._extract_semantic_info(messages)
        if not semantic_info:
            raise ValueError("No valid concepts found in the input messages")
        
        questions = []
        used_concepts = set()
        attempts = 0
        max_attempts = num_questions * 2  # Allow some extra attempts to find valid questions
        
        while len(questions) < num_questions and attempts < max_attempts:
            # Select a random message's semantic info
            info = random.choice(semantic_info)
            
            # Get available concepts that haven't been used
            available_concepts = [c for c in info['concepts'] if c['term'] not in used_concepts]
            
            # If all concepts have been used, reset the used concepts set
            if not available_concepts:
                used_concepts.clear()
                available_concepts = info['concepts']
            
            if available_concepts:
                # Select complexity level based on weights
                complexity = random.choices(
                    list(self.complexity_weights.keys()),
                    weights=list(self.complexity_weights.values())
                )[0]
                
                # Generate question
                concept_info = random.choice(available_concepts)
                question = self._generate_mcq(concept_info, complexity)
                
                if question:
                    used_concepts.add(question['concept'])
                    questions.append(question)
            
            attempts += 1
        
        if not questions:
            raise ValueError("Failed to generate any valid questions")
        
        return questions



def displayAssignment(questions):
    """Display MCQ assignment with improved UI and feedback."""
    answers = {}
    st.header("MCQ Quiz")
    
    # Display the stats in a more appealing format
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", st.session_state["NumQuestions"])
    with col2:
        st.metric("Attempted", st.session_state["Attempted"])
    with col3:
        st.metric("Correct", st.session_state["Correct"])
    with col4:
        st.metric("Wrong", st.session_state["Wrong"])
    
    st.divider()
    
    # Display the questions with improved formatting
    for index, question in enumerate(questions):
        with st.container():
            st.subheader(f"Question {index + 1}")
            st.markdown(f"**{question['question']}**")
            st.markdown(f"*Complexity: {question['complexity'].title()}*")
            
            # Format options with proper spacing and numbering
            formatted_options = [f"{opt}" for opt in question["options"]]
            
            r = st.radio(
                "Select your answer:",
                formatted_options,
                index=None,
                key=f"q{index}"
            )
            
            if f"answered_q{index}" not in st.session_state:
                st.session_state[f"answered_q{index}"] = False

            if r and not st.session_state[f"answered_q{index}"]:
                st.session_state[f"answered_q{index}"] = True
                st.session_state["Attempted"] += 1
                
                if r == question["correct_answer"]:
                    st.success("✔️ Correct!", icon="✅")
                    st.session_state["Correct"] += 1
                else:
                    st.error("❌ Wrong! The correct answer is:", icon="❌")
                    st.info(question["correct_answer"])
                    st.session_state["Wrong"] += 1
            
            answers[f"q{index}"] = r
            st.divider()
    
    st.session_state["Answers"] = answers
    
    if st.session_state["Attempted"] == st.session_state["NumQuestions"]:
        score_percentage = (st.session_state["Correct"] / st.session_state["NumQuestions"]) * 100
        st.balloons()
        st.success(f"Quiz completed! Your score: {score_percentage:.1f}%")

def init_session_variables():
    """Initialize or reset session state variables."""
    if "NumQuestions" not in st.session_state:
        st.session_state["NumQuestions"] = 0
    if "Attempted" not in st.session_state:
        st.session_state["Attempted"] = 0
    if "Correct" not in st.session_state:
        st.session_state["Correct"] = 0
    if "Wrong" not in st.session_state:
        st.session_state["Wrong"] = 0
    if "Submitted" not in st.session_state:
        st.session_state["Submitted"] = False
    if "Answers" not in st.session_state:
        st.session_state["Answers"] = {}
    if "Questions" not in st.session_state:
        st.session_state["Questions"] = None

def reset_quiz():
    """Reset all quiz-related session variables."""
    st.session_state["NumQuestions"] = 0
    st.session_state["Attempted"] = 0
    st.session_state["Correct"] = 0
    st.session_state["Wrong"] = 0
    st.session_state["Submitted"] = False
    st.session_state["Answers"] = {}
    st.session_state["Questions"] = None
    
    # Clear all answered question flags
    keys_to_clear = [key for key in st.session_state.keys() if key.startswith("answered_q")]
    for key in keys_to_clear:
        del st.session_state[key]

def init_session_variables():
    """Initialize or reset session state variables."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "friend1_name" not in st.session_state:
        st.session_state["friend1_name"] = ""
    if "friend2_name" not in st.session_state:
        st.session_state["friend2_name"] = ""
    if "current_speaker" not in st.session_state:
        st.session_state["current_speaker"] = None
    if "chat_started" not in st.session_state:
        st.session_state["chat_started"] = False
    if "NumQuestions" not in st.session_state:
        st.session_state["NumQuestions"] = 0
    if "Attempted" not in st.session_state:
        st.session_state["Attempted"] = 0
    if "Correct" not in st.session_state:
        st.session_state["Correct"] = 0
    if "Wrong" not in st.session_state:
        st.session_state["Wrong"] = 0
    if "Questions" not in st.session_state:
        st.session_state["Questions"] = None

def reset_chat():
    """Reset all chat-related session variables."""
    st.session_state["messages"] = []
    st.session_state["current_speaker"] = None
    st.session_state["chat_started"] = False
    st.session_state["NumQuestions"] = 0
    st.session_state["Attempted"] = 0
    st.session_state["Correct"] = 0
    st.session_state["Wrong"] = 0
    st.session_state["Questions"] = None

def display_chat():
    """Display the chat messages with black and red color scheme."""
    st.markdown("""
        <style>
        .chat-message-friend1 {
            background-color: #2F2F2F;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
        }
        .chat-message-friend2 {
            background-color: #8B0000;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
        }
        .timestamp {
            color: #CCCCCC;
            font-size: 0.8em;
        }
        .sender-name {
            font-weight: bold;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("Chat History")
    for msg in st.session_state["messages"]:
        message_class = "chat-message-friend1" if msg["sender"] == st.session_state["friend1_name"] else "chat-message-friend2"
        st.markdown(
            f'<div class="{message_class}">'
            f'<span class="sender-name">{msg["sender"]}</span>: {msg["message"]}<br>'
            f'<span class="timestamp">{msg["timestamp"]}</span>'
            f'</div>', 
            unsafe_allow_html=True
        )

def displayAssignment(questions):
    """Display MCQ assignment with improved UI and feedback."""
    answers = {}
    st.header("MCQ Quiz")
    
    # Display the stats in a more appealing format
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", st.session_state["NumQuestions"])
    with col2:
        st.metric("Attempted", st.session_state["Attempted"])
    with col3:
        st.metric("Correct", st.session_state["Correct"])
    with col4:
        st.metric("Wrong", st.session_state["Wrong"])
    
    st.divider()
    
    # Display the questions with improved formatting
    for index, question in enumerate(questions):
        with st.container():
            st.subheader(f"Question {index + 1}")
            st.markdown(f"**{question['question']}**")
            st.markdown(f"*Complexity: {question['complexity'].title()}*")
            
            formatted_options = [f"{opt}" for opt in question["options"]]
            
            r = st.radio(
                "Select your answer:",
                formatted_options,
                index=None,
                key=f"q{index}"
            )
            
            if f"answered_q{index}" not in st.session_state:
                st.session_state[f"answered_q{index}"] = False

            if r and not st.session_state[f"answered_q{index}"]:
                st.session_state[f"answered_q{index}"] = True
                st.session_state["Attempted"] += 1
                
                if r == question["correct_answer"]:
                    st.success("✔️ Correct!", icon="✅")
                    st.session_state["Correct"] += 1
                else:
                    st.error("❌ Wrong! The correct answer is:", icon="❌")
                    st.info(question["correct_answer"])
                    st.session_state["Wrong"] += 1
            
            st.divider()

def main():
    st.set_page_config(page_title="Chat MCQ Generator", page_icon="💭", layout="wide")
    
    st.title("💭 Friend Chat & MCQ Generator")
    st.markdown("""
    Have a conversation with your friend and generate MCQs based on your chat!
    The questions will be generated from your last 10 messages.
    """)
    
    init_session_variables()
    
    # Get friends' names if not already set
    if not st.session_state["chat_started"]:
        col1, col2 = st.columns(2)
        with col1:
            friend1_name = st.text_input("Enter first friend's name (black messages):", key="friend1")
        with col2:
            friend2_name = st.text_input("Enter second friend's name (red messages):", key="friend2")
        
        if st.button("Start Chat") and friend1_name and friend2_name:
            st.session_state["friend1_name"] = friend1_name
            st.session_state["friend2_name"] = friend2_name
            st.session_state["current_speaker"] = friend1_name
            st.session_state["chat_started"] = True
            st.rerun()
    
    # Chat interface
    if st.session_state["chat_started"]:
        display_chat()
        
        # Chat input
        with st.container():
            st.write(f"Current speaker: **{st.session_state['current_speaker']}**")
            col1, col2 = st.columns([4, 1])
            
            with col1:
                message = st.text_input("Type your message:", key="message_input")
            
            with col2:
                send_btn = st.button("Send")
                
            if send_btn and message:
                # Add message to chat history
                st.session_state["messages"].append({
                    "sender": st.session_state["current_speaker"],
                    "message": message,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Switch speakers
                st.session_state["current_speaker"] = (
                    st.session_state["friend2_name"] 
                    if st.session_state["current_speaker"] == st.session_state["friend1_name"] 
                    else st.session_state["friend1_name"]
                )
                st.rerun()
        
        # Generate MCQ button
        if len(st.session_state["messages"]) >= 5:
            col1, col2 = st.columns([4, 1])
            with col1:
                num_questions = st.number_input(
                    'Number of MCQs to generate',
                    min_value=1,
                    max_value=10,
                    value=5
                )
            with col2:
                if st.button("Generate MCQs"):
                    # Get last 10 messages or all if less than 10
                    last_messages = [
                        msg["message"] 
                        for msg in st.session_state["messages"][-10:]
                    ]
                    
                    with st.spinner("Generating questions..."):
                        try:
                            analyzer = ChatAnalyzer()
                            st.session_state["Questions"] = analyzer.generate_mcqs(
                                last_messages, 
                                num_questions
                            )
                            st.session_state["NumQuestions"] = num_questions
                            st.success("Questions generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating questions: {str(e)}")
        
        # Display MCQs if generated
        if st.session_state["Questions"]:
            displayAssignment(st.session_state["Questions"])
        
        # Reset button
        if st.button("Reset Chat", type="secondary"):
            reset_chat()
            st.rerun()

if __name__ == "__main__":
    main()