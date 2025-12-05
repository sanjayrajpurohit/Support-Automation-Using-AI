from flask import Flask, request, render_template, session, redirect, url_for, send_from_directory, send_file
from transformers import pipeline
from pypdf import PdfReader
from docx import Document
from io import BytesIO
import os
import csv
import pandas as pd
import re
from datetime import datetime, timedelta

DOCUMENT_FOLDER = os.path.join(os.path.dirname(__file__), "SOP")
MODEL_NAME = "google/flan-t5-base"
LOG_FILE = "chat_logs.csv"

app = Flask(__name__)
app.secret_key = "Support_Automation"
qa_pipeline = None

def extract_text_from_file(file_path):
    try:
        with open(file_path, "rb") as f:
            buf = BytesIO(f.read())
        if file_path.lower().endswith(".pdf"):
            pdf_reader = PdfReader(buf)
            return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file_path.lower().endswith(".docx"):
            doc = Document(buf)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

def chunk_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def extract_cause_section(text):
    """Extract only the cause/issue description from document text, excluding solution steps"""
    
    # First, try to find text before "Solution Steps" or similar markers
    solution_markers = [
        r"solution steps?:?",
        r"steps to be performed:?",
        r"steps:?",
        r"solution:?",
        r"•\s*open",
        r"•\s*take",
        r"•\s*copy",
        r"•\s*perform",
        r"1\.\s*open",
        r"1\.\s*take"
    ]
    
    text_before_solution = text
    for marker in solution_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text_before_solution = text[:match.start()].strip()
            break
    
    # Look for cause descriptions after "How the issue occurs?" or similar
    cause_patterns = [
        r"how the issue occurs?\s*(.+?)(?=solution|steps|•|$)",
        r"overview[:\s]*(.+?)(?=solution|steps|•|$)",
        r"issue occurred if\s*(.+?)(?=solution|steps|•|$)"
    ]
    
    cause_text = ""
    for pattern in cause_patterns:
        match = re.search(pattern, text_before_solution, re.IGNORECASE | re.DOTALL)
        if match:
            cause_text = match.group(1).strip()
            break
    
    # If no specific pattern found, use the text before solution steps
    if not cause_text and text_before_solution:
        # Remove title and keep the descriptive part
        lines = text_before_solution.split('\n')
        desc_lines = []
        for line in lines[1:]:  # Skip first line (usually title)
            line = line.strip()
            if line and not any(skip_word in line.lower() for skip_word in 
                              ["how the issue occurs?", "solution", "steps"]):
                desc_lines.append(line)
        cause_text = ' '.join(desc_lines)
    
    # Clean up and filter the cause text
    if cause_text:
        # Remove solution-related sentences
        sentences = [s.strip() for s in cause_text.split('.') if s.strip()]
        cause_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Skip sentences with action words (solution steps)
            if not any(action in sentence_lower for action in 
                      ["open", "click", "perform", "run", "save", "copy", "paste", "export", 
                       "filter", "update", "calculate", "select", "enter", "navigate"]):
                # Keep sentences that describe conditions or causes
                if (len(sentence) > 10 and 
                    any(indicator in sentence_lower for indicator in 
                       ["occur", "if", "when", "because", "due to", "caused", "happens", 
                        "lease", "date", "accounting", "month", "day", "schedule", "payment",
                        "overview", "issue", "problem"])):
                    cause_sentences.append(sentence)
        
        if cause_sentences:
            return '. '.join(cause_sentences).strip() + '.'
    
    return ""

def get_available_documents():
    files = []
    for filename in os.listdir(DOCUMENT_FOLDER):
        # Skip temporary files (starting with ~$) and hidden files
        if (filename.lower().endswith((".pdf", ".docx")) and 
            not filename.startswith("~$") and 
            not filename.startswith(".")):
            files.append(filename)
    return files

def find_relevant_document(user_issue):
    """Find the most relevant document based on user's issue description"""
    documents = get_available_documents()
    
    if not documents:
        return None
    
    user_issue_lower = user_issue.lower()
    user_words = set(user_issue_lower.split())
    
    # Remove common stop words that don't help with matching
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was', 'with', 'for'}
    user_words = user_words - stop_words
    
    document_scores = {}
    
    print(f"Available documents: {documents}")
    print(f"User issue: {user_issue}")
    print(f"User words (filtered): {user_words}")
    
    for doc in documents:
        score = 0
        doc_name_lower = doc.lower()
        doc_name_without_ext = doc_name_lower.rsplit('.', 1)[0]  # Remove file extension
        doc_words = set(doc_name_without_ext.replace('_', ' ').replace('-', ' ').split())
        
        # Direct word matches in document name (highest priority)
        direct_matches = user_words.intersection(doc_words)
        score += len(direct_matches) * 10
        
        # Partial word matches in document name
        for user_word in user_words:
            for doc_word in doc_words:
                if len(user_word) > 3 and user_word in doc_word:
                    score += 5
                elif len(doc_word) > 3 and doc_word in user_word:
                    score += 5
        
        # Check if user words appear anywhere in the document name
        for user_word in user_words:
            if len(user_word) > 2 and user_word in doc_name_lower:
                score += 3
        
        # Bonus scoring for common issue types found in document names
        issue_indicators = {
            'login': ['login', 'signin', 'sign', 'access', 'auth'],
            'password': ['password', 'pwd', 'pass', 'reset'],
            'email': ['email', 'mail', 'outlook', 'exchange', 'smtp'],
            'network': ['network', 'wifi', 'internet', 'connection', 'vpn'],
            'printer': ['printer', 'print', 'printing'],
            'software': ['software', 'app', 'application', 'install', 'setup'],
            'error': ['error', 'troubleshoot', 'fix', 'problem', 'issue'],
            'server': ['server', 'service', 'database', 'db'],
            'security': ['security', 'virus', 'malware', 'firewall'],
            'backup': ['backup', 'restore', 'recovery']
        }
        
        for category, keywords in issue_indicators.items():
            if any(keyword in user_issue_lower for keyword in keywords):
                if any(keyword in doc_name_lower for keyword in keywords):
                    score += 8
                elif category in doc_name_lower:
                    score += 6
        
        # Try to analyze document content for better matching (first 1000 characters)
        try:
            file_path = os.path.join(DOCUMENT_FOLDER, doc)
            content_sample = extract_text_from_file(file_path)[:1000].lower()
            
            # Check if user words appear in document content
            for user_word in user_words:
                if len(user_word) > 3 and user_word in content_sample:
                    score += 2
        except:
            pass  # Skip content analysis if file can't be read
        
        document_scores[doc] = score
        print(f"Document: {doc} - Score: {score} - Doc words: {doc_words}")
    
    # Return the document with highest score
    if document_scores and max(document_scores.values()) > 0:
        best_doc = max(document_scores.items(), key=lambda x: x[1])[0]
        print(f"Selected document: {best_doc} with score: {document_scores[best_doc]}")
        return best_doc
    
    # If no good matches found, return the first document
    print(f"No good matches found, returning first document: {documents[0]}")
    return documents[0]

def log_interaction(question, answer, ip_address, document_name):
    timestamp = datetime.now().isoformat()
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(["timestamp", "ip_address", "question", "answer", "document"])
        writer.writerow([timestamp, ip_address, question, answer, document_name])


@app.route("/", methods=["GET", "POST"])
def home():
    """Main route that handles initial issue input and document selection"""
    if request.method == "POST":
        user_issue = request.form.get("issue", "").strip()
        
        if not user_issue:
            return render_template("chat.html", 
                                 history=[], 
                                 selected_document="",
                                 show_issue_form=True,
                                 error="Please describe your issue.")
        
        # Find the most relevant document based on the issue
        relevant_doc = find_relevant_document(user_issue)
        
        if not relevant_doc:
            return render_template("chat.html", 
                                 history=[], 
                                 selected_document="",
                                 show_issue_form=True,
                                 error="No documents available.")
        
        # Load the selected document
        file_path = os.path.join(DOCUMENT_FOLDER, relevant_doc)
        context_text = extract_text_from_file(file_path)
        
        if not context_text.strip():
            return render_template("chat.html", 
                                 history=[], 
                                 selected_document="",
                                 show_issue_form=True,
                                 error="Could not extract text from the selected document.")
        
        # Store in session
        session["selected_document"] = relevant_doc
        session["context_text"] = context_text
        session["initial_issue"] = user_issue
        session["history"] = []
        
        # Generate initial response based on the issue
        initial_response = f"I understand you're experiencing: '{user_issue}'. I've selected the most relevant documentation ({relevant_doc}) to help you. How can I assist you further with this issue?"
        
        session["history"].append({
            "question": user_issue, 
            "answer": initial_response
        })
        
        # Log the initial interaction
        ip_address = request.remote_addr
        log_interaction(user_issue, initial_response, ip_address, relevant_doc)
        
        return render_template("chat.html", 
                             history=session["history"], 
                             selected_document=relevant_doc,
                             show_issue_form=False)
    
    # GET request - show the issue input form
    return render_template("chat.html", 
                         history=[], 
                         selected_document="",
                         show_issue_form=True)

@app.route("/select_document", methods=["GET", "POST"])
def select_document():
    """Route for manual document selection (optional)"""
    documents = get_available_documents()

    if request.method == "POST":
        selected_doc = request.form.get("document")
        if selected_doc not in documents:
            return "Invalid document selected", 400

        file_path = os.path.join(DOCUMENT_FOLDER, selected_doc)
        context_text = extract_text_from_file(file_path)

        if not context_text.strip():
            return "No text extracted from selected document.", 400

        session["selected_document"] = selected_doc
        session["context_text"] = context_text
        session["history"] = []

        return redirect(url_for("chat"))

    return render_template("select_document.html", documents=documents, selected=session.get("selected_document"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    """Chat route for ongoing conversation"""
    global qa_pipeline

    if "context_text" not in session:
        return redirect(url_for("home"))

    context_text = session["context_text"]

    if "history" not in session:
        session["history"] = []

    answer = ""
    user_question = ""

    if request.method == "POST":
        user_question = request.form["question"].strip()
        lower_q = user_question.lower()

        # Handle greetings separately
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(lower_q == g for g in greetings):
            answer = "Hello! How can I help you today?"
        else:
            chunks = chunk_text(context_text, max_length=500)[:5]
            answers = []

            # Determine if user is asking for cause/description or solution
            # More specific phrase detection for causes
            cause_phrases = [
                "how does", "how the", "how this", "why does", "why is", "why do", "why the", 
                "what causes", "what is causing", "what makes", "what leads to",
                "how occurs", "how occur", "how happens", "how happened",
                "reason for", "cause of", "explain why", "explain how",
                "what is the reason", "what is the cause"
            ]
            
            solution_phrases = [
                "how to fix", "how to solve", "how to resolve", "how can i fix", "how can i solve",
                "fix this", "solve this", "resolve this", "repair this",
                "troubleshoot", "solution", "steps to", "procedure to", "guide to",
                "how do i fix", "what should i do", "help me fix", "help me solve"
            ]
            
            # Check for exact phrases first
            is_asking_for_cause = any(phrase in lower_q for phrase in cause_phrases)
            is_asking_for_solution = any(phrase in lower_q for phrase in solution_phrases)
            
            # Additional check: if question starts with "how" but contains "occur/occurs/happen", it's likely asking for cause
            if lower_q.startswith("how") and any(word in lower_q for word in ["occur", "occurs", "happen", "happens", "work", "works"]):
                is_asking_for_cause = True
                is_asking_for_solution = False
            
            # If still unclear, use word counting as fallback
            if not is_asking_for_cause and not is_asking_for_solution:
                cause_indicators = ["why", "what", "cause", "reason", "explain", "describe", "occurs", "happen"]
                solution_indicators = ["fix", "solve", "resolve", "repair", "help"]
                
                cause_count = sum(1 for word in cause_indicators if word in lower_q)
                solution_count = sum(1 for word in solution_indicators if word in lower_q)
                
                if cause_count > solution_count and cause_count > 0:
                    is_asking_for_cause = True
                elif solution_count > 0:
                    is_asking_for_solution = True

            try:
                # For cause questions, first try to extract cause section directly from document
                if is_asking_for_cause:
                    cause_section = extract_cause_section(context_text)
                    if cause_section:
                        answer = cause_section
                        answers = [answer]  # Use the extracted cause as the final answer
                    else:
                        # Fallback to model generation with focused prompt
                        answers = []
                        for chunk in chunks[:2]:  # Use fewer chunks for cause questions
                            prompt = f"""Extract only the explanation of what causes this issue from the document below. 
Do not include any solution steps or procedures.

Document:
{chunk}

Question: {user_question}

Cause:"""
                            
                            result = qa_pipeline(prompt, max_new_tokens=80)[0]["generated_text"]
                            clean_result = result.strip()
                            
                            # Remove prompt echoes
                            unwanted = ["Extract only", "Do not include", "Document:", "Question:", "Cause:", 
                                       "explanation of what causes", "from the document below"]
                            for phrase in unwanted:
                                clean_result = clean_result.replace(phrase, "")
                            
                            # Remove solution-related content
                            sentences = [s.strip() for s in clean_result.split('.') if s.strip()]
                            cause_sentences = []
                            for sentence in sentences:
                                if (len(sentence) > 5 and 
                                    not any(action in sentence.lower() for action in 
                                           ["open", "click", "perform", "run", "save", "copy", "export", "update"])):
                                    cause_sentences.append(sentence)
                            
                            if cause_sentences:
                                clean_result = '. '.join(cause_sentences[:2]).strip()  # Max 2 sentences
                                if clean_result and not clean_result.endswith('.'):
                                    clean_result += '.'
                                if clean_result:
                                    answers.append(clean_result)
                                break
                else:
                    # For solution questions or general questions
                    for chunk in chunks:
                        if is_asking_for_solution:
                            # User wants solution/resolution steps
                            prompt = f"""You are a support assistant. The user wants to know HOW TO FIX or RESOLVE an issue.
Provide ONLY step-by-step solution or troubleshooting procedures from the document below.
Do NOT explain what causes the issue or describe the problem.

Document:
{chunk}

User Question: {user_question}
Solution Steps:"""
                        elif is_asking_for_cause:
                            # User wants to know cause/description only
                            prompt = f"""Based on this document, explain what causes this issue in 1-2 sentences:

{chunk}

Question: {user_question}

The issue occurs when:"""
                        else:
                            # General question - provide brief overview
                            prompt = f"""You are a support assistant. Answer the user's question based on the document below.
Provide a concise and relevant answer.

Document:
{chunk}

User Question: {user_question}
Answer:"""
                        
                        result = qa_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]
                        clean_result = result.strip()
                    
                    # Remove repetitive patterns first
                    # Check for repetitive phrases (same phrase repeated multiple times)
                    words = clean_result.split()
                    if len(words) > 0:
                        # Find if there's a repeating pattern
                        for phrase_length in range(2, 6):  # Check for 2-5 word repeating phrases
                            if len(words) >= phrase_length * 3:  # Need at least 3 repetitions to detect
                                phrase = ' '.join(words[:phrase_length])
                                # Count how many times this phrase appears consecutively
                                count = 0
                                for i in range(0, len(words) - phrase_length + 1, phrase_length):
                                    if ' '.join(words[i:i+phrase_length]) == phrase:
                                        count += 1
                                    else:
                                        break
                                
                                # If phrase repeats more than 2 times, keep only one
                                if count > 2:
                                    clean_result = phrase + '.'
                                    break
                    
                    # Comprehensive list of instruction echoes and unwanted text to remove
                    unwanted_phrases = [
                        "You are a support assistant", "The user wants to know", "Provide ONLY", 
                        "Do NOT explain", "STRICTLY AVOID", "Focus only on", "User Question:",
                        "Solution Steps:", "Cause/Explanation", "NO SOLUTION STEPS", 
                        "Document:", "Answer:", "Cause Description:", "Question:", "Based on this document",
                        "explain what causes", "in 1-2 sentences", "The issue occurs when:",
                        "step-by-step solution or troubleshooting procedures from the document below",
                        "the cause, description, or explanation of the problem",
                        "what causes the issue or describe the problem",
                        "HOW TO FIX or RESOLVE an issue",
                        "troubleshooting procedures from the document below",
                        "step-by-step solution or troubleshooting procedures"
                    ]
                    
                    # Remove unwanted phrases (case insensitive)
                    for phrase in unwanted_phrases:
                        # Remove exact matches
                        clean_result = clean_result.replace(phrase, "")
                        # Remove case insensitive matches
                        clean_result = re.sub(re.escape(phrase), "", clean_result, flags=re.IGNORECASE)
                    
                    # Split into sentences and filter more aggressively
                    sentences = []
                    for sentence in clean_result.split('.'):
                        sentence = sentence.strip()
                        if sentence:
                            # Skip sentences that contain any unwanted phrases
                            if not any(phrase.lower() in sentence.lower() for phrase in unwanted_phrases):
                                # Skip very short fragments that are likely artifacts
                                if len(sentence.split()) > 2:
                                    sentences.append(sentence)
                    
                    clean_result = '. '.join(sentences).strip()
                    if clean_result and not clean_result.endswith('.'):
                        clean_result += '.'
                    
                    # Specific filtering based on question type
                    if is_asking_for_cause:
                        # For cause questions, remove any solution steps
                        solution_keywords = [
                            "1.", "2.", "3.", "step", "perform", "uncheck", "return", 
                            "recalculate", "submit", "click", "navigate", "select", "open",
                            "copy", "paste", "take", "export", "run", "save", "update",
                            "• open", "• take", "• copy", "• perform"
                        ]
                        
                        sentences = clean_result.split('.')
                        cause_sentences = []
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence and len(sentence) > 10:  # Minimum length check
                                # Skip if contains solution keywords
                                has_solution_keyword = any(keyword.lower() in sentence.lower() for keyword in solution_keywords)
                                # Skip if starts with numbers or bullets
                                starts_with_step = sentence.strip().startswith(('1', '2', '3', '4', '5', '•'))
                                
                                if not has_solution_keyword and not starts_with_step:
                                    # Include if it describes the cause/issue
                                    cause_indicators = ["occur", "happens", "caused", "issue", "lease", "date", "accounting", "overview"]
                                    if any(indicator.lower() in sentence.lower() for indicator in cause_indicators):
                                        cause_sentences.append(sentence)
                        
                        clean_result = '. '.join(cause_sentences).strip()
                        if clean_result and not clean_result.endswith('.'):
                            clean_result += '.'
                            
                    elif is_asking_for_solution:
                        # For solution questions, keep only actionable steps
                        cause_keywords = [
                            "was changed", "occurred because", "happens when", "the issue is caused",
                            "analyzed the lease found", "gain/loss was calculated"
                        ]
                        sentences = clean_result.split('.')
                        solution_sentences = []
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence and not any(keyword in sentence.lower() for keyword in cause_keywords):
                                solution_sentences.append(sentence)
                        
                        clean_result = '. '.join(solution_sentences).strip()
                        if clean_result and not clean_result.endswith('.'):
                            clean_result += '.'
                    
                    if clean_result:
                        answers.append(clean_result)

                answer = " ".join(answers)
                
                # Skip heavy filtering for cause questions if we used extract_cause_section
                if not (is_asking_for_cause and extract_cause_section(context_text)):
                    # Final cleanup to remove any remaining unwanted text
                    final_unwanted = [
                        "the cause, description, or explanation of the problem",
                        "what causes the issue or describe the problem",
                        "step-by-step solution or troubleshooting procedures",
                        "focus only on the cause", 
                        "strictly avoid",
                        "provide only"
                    ]
                    
                    for unwanted in final_unwanted:
                        answer = re.sub(re.escape(unwanted), "", answer, flags=re.IGNORECASE)
                
                # Remove extra spaces and ensure proper sentence structure
                answer = re.sub(r'\s+', ' ', answer).strip()
                if answer and not answer.endswith('.'):
                    answer += '.'
                
                # Remove leading/trailing periods or spaces
                answer = answer.strip(' .')
                if answer and not answer.endswith('.'):
                    answer += '.'
                    
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        # Store history
        session["history"].append({"question": user_question, "answer": answer})
        session.modified = True

        # Log interaction with updated answer
        ip_address = request.remote_addr
        log_interaction(user_question, answer, ip_address, session.get("selected_document", "Unknown"))

    return render_template("chat.html", 
                         history=session["history"], 
                         selected_document=session.get("selected_document", ""),
                         show_issue_form=False)

@app.route("/download_report")
def download_report():
    if not os.path.exists(LOG_FILE):
        return "No logs available", 404

    try:
        df = pd.read_csv(LOG_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        one_week_ago = datetime.now() - timedelta(days=7)
        recent_df = df[df["timestamp"] >= one_week_ago]

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            recent_df.to_excel(writer, index=False, sheet_name="Weekly Report")

        output.seek(0)
        filename = f"weekly_chat_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        return send_file(output, as_attachment=True, download_name=filename, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        return f"Error generating report: {e}", 500

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    """Reset chat and return to initial issue form"""
    session.clear()
    return redirect(url_for("home"))

@app.route("/new_issue")
def new_issue():
    """Start a new issue - clear session and redirect to home"""
    session.clear()
    return redirect(url_for("home"))

def init_chatbot():
    global qa_pipeline
    print("🤖 Loading FLAN-T5 model...")
    qa_pipeline = pipeline("text2text-generation", model=MODEL_NAME)

if __name__ == "__main__":
    init_chatbot()
    app.run(debug=True)
