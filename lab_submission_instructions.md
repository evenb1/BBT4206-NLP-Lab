# Lab Submission: Business Intelligence Course Evaluation NLP Analysis

---

## Student Details

**Name of the team on GitHub Classroom:** B8

**Team Member Contributions:**

**Member 1**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | 152508      |
| **Name:**                                                                                          | Andrew M    |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Data loading, exploration, and preprocessing. Created the initial notebook structure, handled CSV data import, performed exploratory data analysis on 130+ course evaluations, and prepared the dataset for NLP analysis. Learned about the importance of thorough data exploration and proper text data handling in NLP projects.        |

**Member 2**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | 146793      |
| **Name:**                                                                                          | Even Russom |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Topic modeling implementation using LDA. Designed and implemented the Latent Dirichlet Allocation model, created the document-term matrix using CountVectorizer, trained the 5-topic model, and extracted topic labels. Learned how LDA discovers latent themes in unstructured text data and how to interpret topic coherence.        |

**Member 3**

| **Details**                                                                                        | **Comment** |
|:---------------------------------------------------------------------------------------------------|:------------|
| **Student ID:**                                                                                    | 147120      |
| **Name:**                                                                                          | Ojijo Josh  |
| **What part of the lab did you personally contribute to,** <br>**and what did you learn from it?** | Sentiment analysis and Streamlit app development. Implemented VADER sentiment analysis, trained Naive Bayes sentiment classifier, created the Streamlit web interface with real-time prediction capabilities, and deployed to Streamlit Cloud. Learned how to build production-ready NLP applications and deploy machine learning models for end-users.      |

---

## Scenario

Your client, a university, is seeking to enhance their qualitative analysis of student course evaluations collected from students. They have provided a dataset containing student course evaluations for two courses in the Business Intelligence Option (BBT 4106: Business Intelligence I and BBT 4206: Business Intelligence II). 

Using Natural Language Processing (NLP) techniques, we identified the key topics (themes) discussed in course evaluations and determined the sentiments (positive, negative, neutral) associated with each theme. We also created an interactive web interface enabling the university to input new evaluation text and receive real-time topic and sentiment predictions.

---

## Key Findings

### Topics Discovered (5 Major Themes)

**Topic 0: Course Content & Lecture Materials**
- Key terms: liked, lab, practical, slide, content, topic, number, data, note, time, reduce, module, lecturer, learning, teaching
- **Interpretation:** This topic focuses on students' feedback regarding the volume and presentation of course materials. The prominent word "reduce" suggests students find the slides and notes overwhelming. This is the most discussed theme in evaluations.

**Topic 1: Assessment & Learning Depth**
- Key terms: cat, point, end, note, learning, maybe, depth, quiz, covered, year, lecturer, dont, sure, issue, understand
- **Interpretation:** Students discuss continuous assessment tests (CATs), end-of-module quizzes, and their concerns about whether topics are covered in sufficient depth. The uncertainty words ("maybe," "dont," "sure") suggest confusion about assessment expectations.

**Topic 2: Laboratory Work & Practical Application**
- Key terms: lab, topic, work, concept, slide, understanding, content, module, practical, lecturer, assignment, group, better, application, understand
- **Interpretation:** This topic emphasizes the value students place on practical laboratory exercises. The word "better" suggests students believe lab work improves their understanding compared to theory alone. Group assignments are also discussed positively.

**Topic 3: Business Intelligence Tools & Real-World Application**
- Key terms: data, business, practical, learning, tool, intelligence, project, student, understand, study, case, topic, using, handson, theory
- **Interpretation:** Students appreciate hands-on experience with BI tools and real-world applications. This topic reflects the connection between theoretical learning and practical business scenarios, with emphasis on case studies and hands-on experience.

**Topic 4: Collaborative Learning & Relevance**
- Key terms: group, work, assignment, moment, matter, opinion, far, okay, relevant, practical, data, really, module, helped, research
- **Interpretation:** This topic captures students' views on group work, practical assignments, and overall course relevance. The phrase "really helped" and "relevant" indicate positive reception of collaborative learning approaches.

### Sentiment Distribution

- **Positive:** 87 evaluations (69.0%) 
- **Neutral:** 28 evaluations (22.2%) 
- **Negative:** 11 evaluations (8.7%) 

### Key Correlations & Patterns

1. **Absenteeism Impact:** Correlation between absenteeism and course rating is -0.142
   - Negative correlation indicates that students with higher absenteeism tend to give slightly lower course ratings
   - However, the weak correlation (-0.142) suggests other factors significantly influence satisfaction

2. **Topic-Sentiment Relationship:**
   - Topics 2 and 3 (practical work and real-world application) show predominantly positive sentiment
   - Topic 0 (course materials volume) shows mixed sentiment, with some negative feedback
   - Topic 1 (assessment) shows more neutral/uncertain sentiment

3. **Student Enjoyment:**
   - Students who reported enjoying the course also showed more positive sentiment in their evaluations
   - Students with low enjoyment ratings frequently mentioned content volume and lack of clarity

---

## Interpretation and Recommendation

### Interpretation

The analysis reveals that students have **predominantly positive feedback** about the Business Intelligence courses, with 69% of evaluations expressing positive sentiment. This suggests the overall course design and teaching approach are effective in meeting student expectations.

The five discovered topics indicate that students appreciate the **practical, hands-on components** of the courses (Topics 2, 3, 4), particularly laboratory work, group assignments, and real-world case studies. These topics consistently receive positive sentiment. However, a critical concern emerges in **Topic 0**, where students repeatedly recommend reducing the volume of lecture slides and course materials. Many evaluations suggest that the content density is overwhelming, making it difficult for students to prioritize and understand what to focus on.

The **weak negative correlation between absenteeism and course ratings (-0.142)** suggests that while attendance matters somewhat, course satisfaction is driven more by content delivery and practical engagement than by attendance alone. This indicates that improving teaching methods and material presentation would likely have a greater impact than attendance policies.

The **assessment-related feedback (Topic 1)** reveals uncertainty among students about CAT expectations and course coverage. While this topic shows neutral sentiment rather than strongly negative, the language used ("dont," "sure," "issue") suggests students feel confused about assessment criteria and depth expectations.

### Recommendations

Based on our analysis, we recommend the following evidence-based improvements:

**1. Optimize Lecture Materials (HIGH PRIORITY)**
- **Evidence:** Topic 0 is the most frequently discussed theme, with "reduce" appearing as a key term. 69% positive sentiment overall drops when students discuss materials volume. Multiple evaluations explicitly request shorter notes and fewer slides.
- **Action:** Condense lecture slides to key concepts. Create summary documents highlighting essential points rather than detailed slide decks. Consider providing separate supplementary materials for deeper dives rather than including everything in main slides.
- **Expected Impact:** Reduce student overwhelm, improve retention, and boost satisfaction scores.

**2. Expand Practical & Real-World Components**
- **Evidence:** Topics 2, 3, and 4 (practical work, BI tools, and case studies) consistently receive positive sentiment. Students explicitly mention that practical exercises help them understand concepts better.
- **Action:** Increase the number of hands-on lab exercises aligned with business scenarios. Include more real-world datasets and case studies. Maintain or expand group project components.
- **Expected Impact:** Strengthen learning outcomes and maintain high satisfaction levels in these areas.

**3. Clarify Assessment Expectations**
- **Evidence:** Topic 1 shows language suggesting confusion ("maybe," "dont," "sure," "issue") about CATs and course coverage depth. Students need clarity on what will be assessed and how.
- **Action:** Provide explicit CAT guidelines at the start of each module. Clearly communicate which topics are essential vs. supplementary. Offer sample assessment questions early in the course.
- **Expected Impact:** Reduce student anxiety about assessments and improve performance through clearer expectations.

**4. Maintain Group Work & Collaborative Learning**
- **Evidence:** Topics 2 and 4 show that group work and collaborative assignments receive positive sentiment. Students report learning from peers and appreciating the real-world preparation these activities provide.
- **Action:** Continue emphasizing group-based labs and projects. Ensure adequate time is allocated for group work during class.
- **Expected Impact:** Sustain positive sentiment and develop professional collaboration skills essential for BI careers.

**5. Consider Asynchronous Learning Options (SECONDARY)**
- **Evidence:** The weak correlation between absenteeism and ratings (-0.142) suggests that while attendance is beneficial, students may have legitimate reasons for absence. Alternative engagement methods could help.
- **Action:** Record lectures and provide supplementary video explanations for key concepts. Create discussion forums for asynchronous interaction.
- **Expected Impact:** Improve accessibility without sacrificing learning outcomes for students with attendance challenges.

---

## Deliverables

### 1. Data Preprocessing & Analysis
-  Text data cleaned and preprocessed (130+ evaluations)
-  Tokenization, lemmatization, and stopword removal applied
-  Missing values handled appropriately
-  Exploratory analysis with word frequency and word clouds generated

### 2. Topic Modeling
-  Latent Dirichlet Allocation (LDA) model with 5 topics trained
-  Document-term matrix created using CountVectorizer (200 features)
-  Topics extracted and interpreted with meaningful labels
-  Topic distribution analyzed across student groups

### 3. Sentiment Analysis
-  VADER sentiment analysis applied to all evaluations
-  Sentiment classifier (Naive Bayes) trained on processed text
-  Multi-class classification (positive, neutral, negative) implemented
-  Sentiment-topic relationships analyzed

### 4. Interface & Deployment
-  Streamlit web application developed with real-time prediction
-  User-friendly interface for topic and sentiment input/output
-  Application deployed locally and ready for demonstration
-  Models (LDA, sentiment classifier) saved and integrated

### 5. Video Demonstration
- 4-minute demo video created
- Jupyter notebook analysis results shown
- Streamlit app demonstrated with multiple examples
- Link: https://www.loom.com/share/e843c117d934436cb0c0662edcd82cb7
- Streamlit application: https://bbt4206-nlp-lab-u4djtgnrxklfm4vjq3ssnx.streamlit.app/
### 6. Application Link
-  Streamlit application: [Application deployed and runnable locally with: `streamlit run app.py`]

---

## Technical Details

### Methods Used
- **Text Preprocessing:** NLTK tokenization, lemmatization, stopword removal
- **Topic Modeling:** Latent Dirichlet Allocation (LDA) with 5 topics
- **Sentiment Analysis:** VADER (Valence Aware Dictionary and sEntiment Reasoner) + Naive Bayes
- **Visualization:** Word clouds, frequency charts, sentiment distribution plots
- **Deployment:** Streamlit web framework

### Data Statistics
- Total evaluations: 130
- Valid for analysis: 126
- Average tokens per evaluation: 45
- Vocabulary size: 200+ unique terms

### Model Performance
- Topic coherence: Clear separation between practical work, materials, and assessment themes
- Sentiment accuracy: 69% positive, 22.2% neutral, 8.7% negative
- Real-time prediction: <1 second per evaluation

---

## Conclusion

The NLP analysis successfully identified the key themes in student course evaluations and revealed predominantly positive sentiment. The results provide actionable insights for improving the Business Intelligence courses, with particular emphasis on streamlining course materials, expanding practical components, and clarifying assessment expectations. The interactive Streamlit application enables the university to process future evaluations in real-time, supporting continuous improvement efforts.

---

**Submitted by:** Team B8 (Andrew M, Even Russom, Ojijo Josh)
**Date:** November 2025
**Course:** BBT 4206 - Natural Language Processing Lab
**Institution:** Strathmore University