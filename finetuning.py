"""
import os: Provides a way to use operating system dependent functionality like reading or writing to the file system.
import torch: Imports PyTorch, a deep learning library.
from datasets import Dataset: Imports the Dataset class from the Hugging Face Datasets library for handling datasets.
from transformers import ...: Imports necessary components from the Hugging Face Transformers library:
BartTokenizer: For tokenizing text.
BartForConditionalGeneration: The BART model used for text generation tasks.
Seq2SeqTrainer: A trainer class specifically for sequence-to-sequence models.
Seq2SeqTrainingArguments: Arguments for training sequence-to-sequence models.
TrainerCallback: Allows custom behavior during training.
DataCollatorForSeq2Seq: Handles padding and other collating tasks for sequence-to-sequence models.
from youtube_transcript_api import YouTubeTranscriptApi: Imports the YouTubeTranscriptApi for fetching transcripts from YouTube videos.
import evaluate: Imports the evaluate module for metrics computation.
"""
import os
import torch
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, DataCollatorForSeq2Seq
from youtube_transcript_api import YouTubeTranscriptApi
import evaluate

"""This function extracts the video ID from a YouTube URL. If the URL contains "v=", it splits the URL at "v=" and takes the part after it."""
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1]
    return url

"""This function fetches the transcript for a given YouTube video ID using the YouTubeTranscriptApi and concatenates all the transcript segments into a single string."""
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = ' '.join([item['text'] for item in transcript])
    return transcript_text

# Load data
video_urls = [
    'https://www.youtube.com/watch?v=y2kg3MOk1sY',
    'https://www.youtube.com/watch?v=916GWv2Qs08',
    'https://www.youtube.com/watch?v=PlxWf493en4',
    'https://www.youtube.com/watch?v=Z1RJmh_OqeA',
    'https://www.youtube.com/watch?v=Uszj_k0DGsg',
    'https://www.youtube.com/watch?v=_ZvnD73m40o',
    'https://www.youtube.com/watch?v=tN6oJu2DqCM',
    'https://www.youtube.com/watch?v=O5nskjZ_GoI',
    'https://www.youtube.com/watch?v=zOjov-2OZ0E',
    'https://www.youtube.com/watch?v=xAcTmDO6NTI',
    'https://www.youtube.com/watch?v=k6U-i4gXkLM',
    'https://www.youtube.com/watch?v=nykOeWgQcHM',
    'https://www.youtube.com/watch?v=G2fqAlgmoPo',
    'https://www.youtube.com/watch?v=d0yGdNEWdn0',
    'https://www.youtube.com/watch?v=w-HYZv6HzAs',
    'https://www.youtube.com/watch?v=F4Zu5ZZAG7I',
    'https://www.youtube.com/watch?v=tTb3d5cjSFI',
    'https://www.youtube.com/watch?v=MKlx1DLa9EA',
    'https://www.youtube.com/watch?v=JsC9ZHi79jo',
    'https://www.youtube.com/watch?v=V1xt7zgnuK0',
    'https://www.youtube.com/watch?v=7XFLTDQ4JMk',
    'https://www.youtube.com/watch?v=e4PTvXtz4GM',
    'https://www.youtube.com/watch?v=2Yt6raj-S1M',
    'https://www.youtube.com/watch?v=V6yixyiJcos',
    'https://www.youtube.com/watch?v=Uq-FOOQ1TpE',
    'https://www.youtube.com/watch?v=hqKafI7Amd8',
    'https://www.youtube.com/watch?v=9vJRopau0g0',
    'https://www.youtube.com/watch?v=xNmf-G81Irs',
    'https://www.youtube.com/watch?v=Bp2Fvkt-TRM',


]

"""
video_urls: List of YouTube video URLs.
video_ids: Extracts video IDs from the URLs using the extract_video_id function.
transcripts: Fetches the transcripts for each video ID using the get_transcript function.
"""
video_ids = [extract_video_id(url) for url in video_urls]
transcripts = [get_transcript(video_id) for video_id in video_ids]

# Manually created summaries (example data, you can update this with actual summaries)
summaries = [
    "This video is an introductory course on computer basics, ideal for beginners and those looking to fill in knowledge gaps. Developed by GCFGlobal.org, it covers a wide range of topics, which can be accessed directly through time codes in the description. The course explains what computers are, their various types, and how they function using hardware and software. It discusses different operating systems such as Windows, macOS, Chrome OS, iOS, and Android, and describes the components of personal computers, laptops, and other devices. The video also touches on internet connectivity, cloud storage, computer maintenance, ergonomics, and online safety. It aims to make the user comfortable with computer terminology, setup, and daily usage.",
    "In this HTML Crash Course by Beau Carnes, you'll learn the fundamentals of HTML, the foundational language for creating web pages. HTML, which stands for Hypertext Markup Language, forms the structure of websites, while CSS adds style and JavaScript provides functionality.Carnes explains HTML's role using a house analogy: HTML is the structure, CSS is the appearance, and JavaScript is the functionality. You'll learn how to write HTML using code editors like CodePen and Visual Studio Code. The course covers basic HTML syntax, including elements like headings, paragraphs, and lists, and attributes like 'src' for images and 'href' for links. You'll also learn how to create forms with text inputs, radio buttons, and checkboxes, and how to use semantic elements like 'main' and 'footer' for better structure and accessibility. Practical examples include creating a simple web page with a cat photo app theme and uploading it to a web hosting service like Hostinger. Overall, this crash course provides a solid foundation in HTML, preparing you to explore CSS and JavaScript for more advanced web development.",
    "In this video, you'll learn the fundamentals of HTML, the language used to create web pages. HTML, standing for Hypertext Markup Language, is a markup language that structures web content but doesn't handle logic like programming languages. You'll start by creating a simple local website. First, create an `index.html` file and open it in both a browser and a code editor like Visual Studio Code. HTML is composed of tags, which define the structure and presentation of the content. For example, the `<b>` tag makes text bold. To create a well-structured HTML document, start with a `<!DOCTYPE html>` declaration followed by the `<html>` tag, and then separate the content into the `<head>` and `<body>` sections. The head contains meta information, while the body includes the visible content.You'll explore various HTML tags, such as `<h1>` to `<h6>` for headings, `<p>` for paragraphs, `<a>` for links, and `<img>` for images. Lists can be created using `<ul>` for unordered lists and `<ol>` for ordered lists, with each item in a `<li>` tag. You'll also learn about tables for organizing data, using `<table>`, `<tr>` for rows, `<th>` for headers, and `<td>` for data cells. By the end, you'll have created a basic web page with headings, paragraphs, images, links, lists, and a table. The video emphasizes good practices like nesting and indenting elements for readability. The next steps involve learning CSS and JavaScript to enhance your web development skills.",
    "In this video, Jake Krieger introduces Flask for Python, focusing on creating a basic CRUD application called 'Task Master'. This tutorial assumes prior knowledge of Python, HTML, and CSS. The video begins with setting up the environment, including installing Python, creating a virtual environment, and installing Flask and Flask-SQLAlchemy. Jake demonstrates creating a simple Flask app that displays 'Hello, World' in the browser. Next, Jake covers static content, templates, and template inheritance using Jinja2. He explains how to create a base template and extend it for other pages, and how to link CSS and JavaScript files. The main application, Task Master, is developed by setting up routes, handling form submissions, and managing tasks using a SQLite database. Jake details creating, reading, updating, and deleting tasks in the application. He walks through creating models, querying the database, and using forms for user input. The video also covers deploying the Flask application to Heroku, including setting up a Procfile and using Git for version control.By the end of the video, viewers will have a functional CRUD app deployed on Heroku, with a clear understanding of Flask basics, template inheritance, static content, and database management.",
    "In this intermediate Git course, Tobias Günther helps improve your Git workflow beyond basics. He covers essential concepts like crafting clear commits, choosing branching strategies, and resolving merge conflicts. Tobias emphasizes creating focused commits and using the staging area effectively. He discusses branching strategies, highlighting the importance of team conventions, and explains the difference between long-running and short-lived branches. He contrasts GitHub Flow, a simple model with one main branch, and Git Flow, which uses multiple branches for features, releases, and development stages. Tobias introduces pull requests for code review and contribution in open-source projects, demonstrating how to create them using GitHub and explaining the role of forks. The course also covers managing merge conflicts, showing how to resolve them manually or with tools, and reassuring that conflicts are manageable and reversible. Finally, Tobias compares merging and rebasing for integrating branches, stressing that rebasing rewrites commit history and should be used carefully. He advises against rewriting commits pushed to shared repositories. Tobias offers a free advanced Git kit with more videos on topics like interactive rebase and submodules to boost Git productivity.",
    "Learn how to get Chat GPT and other LLMs to give you the perfect responses by mastering prompt engineering strategies with Anu Kubo. In this course, she covers the latest techniques to maximize productivity with large language models. Anu explains prompt engineering, AI basics, and LLMs like Chat GPT, along with text-to-image models such as MidJourney. She discusses the importance of prompt engineering, which involves writing and refining prompts to optimize AI interactions, maintaining an up-to-date prompt library, and being a thought leader in this space. The course includes the history of language models from Eliza to GPT-4, the prompt engineering mindset for creating effective prompts, a quick intro to using Chat GPT and its API, and best practices for clear instructions, adopting personas, and specifying formats. It also covers zero-shot and few-shot prompting techniques, understanding AI hallucinations, and vectors and text embeddings for capturing semantic meanings in text. Throughout the course, Anu provides practical examples and emphasizes the importance of clear, specific instructions in crafting prompts to achieve desired AI responses.",
    "In this video, Bo KS outlines the essential technologies needed to become a backend developer. Bo, an experienced course creator for FreeCodeCamp.org, introduces a comprehensive curriculum available on their YouTube channel. A backend developer focuses on server-side logic, managing databases, creating APIs, managing servers, and ensuring security. Key technologies to learn include JavaScript (Node.js), Python, PHP, Java for programming languages, and Git and GitHub for version control. For databases, knowledge of SQL, MySQL, PostgreSQL, and MongoDB is essential. Understanding APIs, specifically RESTful services and GraphQL, is crucial. Caching techniques help reduce load and improve performance, while best practices in security ensure API safety. Testing methodologies, including unit, integration, and end-to-end testing, are vital for maintaining code quality. Knowledge of software design patterns and architectural principles helps create scalable and efficient systems. Familiarity with message brokers like RabbitMQ and Kafka, containerization tools such as Docker and Kubernetes, and web servers like Nginx is important. NoSQL databases like MongoDB and Redis offer flexibility for various data types, while cloud services from AWS, Azure, and Google Cloud enhance scalability and efficiency. Understanding these technologies is crucial for building efficient and secure backend systems. Continuous learning and staying updated with new tools and best practices are essential in this evolving field. Good luck on your learning journey!",
    "The introduction to CrashCourse Computer Science, hosted by Carrie Anne, outlines the series' goal to explore computing topics from basics like bits and logic gates to advanced subjects such as operating systems and virtual reality. The series aims to contextualize computing's role in modern society without teaching programming. Computers, integral to contemporary life, echo past technological revolutions like the Industrial Revolution. Early computing tools, such as the abacus and slide rule, made calculations easier and more accurate, leading to the creation of mechanical devices like the Step Reckoner by Leibniz and Babbage's Difference Engine. Babbage's Analytical Engine, though never built, introduced the concept of a general-purpose computer and inspired early computer scientists, including Ada Lovelace, considered the first programmer. By the late 19th century, computing devices were mainly used in science and engineering. However, Herman Hollerith's electro-mechanical tabulating machine, used in the 1890 US Census, demonstrated computing's efficiency in processing large-scale data, leading to its adoption in business and government. Hollerith's company evolved into IBM, paving the way for the digital computing revolution. The series will delve into these historical developments and their impact on today's world.",
    "In this introductory programming course, Steven and Shawn cover the basics of computer programming applicable to any language through 21 segments. They begin by defining programming as giving precise instructions to a computer, much like guiding a less-than-intelligent friend. Programming languages translate human instructions into machine code, which computers understand. The course focuses on universal programming concepts, avoiding language-specific topics and coding in an IDE. Key topics include variables, basic math, and string manipulation, emphasizing syntax and error handling. Arrays and loops are introduced for managing data and repeating tasks, with explanations of for loops, while loops, and do-while loops. Functions are highlighted as tools for simplifying and reusing code. Error types—syntax, runtime, and logic—are discussed, along with debugging strategies. The course emphasizes frequent testing and debugging to ensure code correctness. By the end, students will understand foundational programming concepts and be prepared to tackle various programming languages and projects.",
    "Ana Bell welcomes the class to 6.100L and introduces the course's focus on Python programming and computational thinking. She emphasizes downloading lecture slides, taking notes, and actively coding during the lectures. The lecture explains the difference between declarative and imperative knowledge, using an algorithm to find the square root of a number. Bell discusses the evolution of computers from fixed-program to stored-program and mentions Alan Turing's contributions. Python's basic primitives—integers, floats, booleans, and NoneType—are introduced, along with the type function and type casting. The importance of meaningful variable names and assignments for better code readability is highlighted. Expressions in Python, operator precedence, and evaluating expressions to a single value are discussed. Examples include a program to calculate the area and circumference of a circle. Bell stresses good code style, using comments and descriptive variable names. She demonstrates using the Python shell and Python Tutor for interactive coding and debugging. The lecture concludes with examples like swapping variable values using a temporary variable, reinforcing key concepts. Bell notes that future lectures will cover decision-making in programs, leading to more complex code execution.",
    "In this introductory lecture, Professor Eric Grimson outlines the structure and goals of MIT's 'Introduction to Computer Science and Programming' course, 6.00. The course, aimed at freshmen and sophomores with little or no prior programming experience, emphasizes computational thinking and the ability to write and understand small pieces of code. The course structure includes two hours of lectures, one hour of recitation, and nine hours of outside work per week, primarily involving programming in Python. The course's strategic goals include preparing students for more advanced computer science courses, instilling confidence in non-majors to read and write code, and providing an understanding of the role and limitations of computation. Professor Grimson stresses the importance of attending lectures and recitations, noting that they will cover material not found in the assigned readings. The lecture also introduces the concepts of declarative and imperative knowledge, highlighting the importance of the latter in computational thinking. Declarative knowledge involves statements of fact, while imperative knowledge provides step-by-step instructions for solving problems. This distinction is fundamental to understanding computation and programming. Professor Grimson explains the evolution from fixed-program computers to stored-program computers, which can execute any sequence of instructions provided as input. He introduces the Python programming language, chosen for its general-purpose, high-level, and interpreted nature. The course will focus on developing good programming practices and understanding the syntax and semantics of Python. The lecture concludes with a brief demonstration of Python, showcasing basic operations and the assignment of values to variables. Students are encouraged to explore Python's capabilities further and to sign up for recitations. This lecture sets the stage for a course that aims to equip students with essential computational skills and knowledge.",
    "In this lecture, Ana Bell introduces the 'Introduction to Computer Science and Programming' course at MIT, providing a comprehensive overview of what students can expect. The course is fast-paced and aims to equip students with programming skills, problem-solving abilities, and a solid understanding of computational concepts using Python. Bell begins with administrative details, emphasizing the importance of attending lectures, downloading materials beforehand, and practicing coding. She explains the foundational concepts of declarative and imperative knowledge, illustrating the difference between knowing facts and knowing how to perform tasks through step-by-step instructions. The lecture covers the basics of computer operations, highlighting that computers perform calculations and store results, and explaining the evolution from fixed-program to stored-program computers. Bell introduces Python as the programming language for the course, detailing its nature as an interpreted, high-level, general-purpose language. She explains how Python manipulates data objects, emphasizing that everything in Python is an object with a specific type that dictates what operations can be performed on it. The lecture covers scalar objects (integers, floats, Booleans, and NoneType) and non-scalar objects, explaining how to check and convert types using Python commands. Bell also discusses the importance of using variables to store values, making code more readable and reusable. She provides examples of variable assignments and explains how variables can be rebound to new values, illustrating with memory diagrams how bindings change but old values remain in memory until garbage collected. The lecture emphasizes the significance of writing readable and well-organized code, highlighting the necessity of good programming practices. Bell concludes by preparing students for upcoming topics, including control flow in Python, which will allow them to write programs that can make decisions and execute different instructions based on those decisions.",
    "In the Introduction to Generative AI course, Dr. Gwendolyn Stripling explains the fundamentals of generative AI, its workings, model types, and applications. Generative AI, a type of artificial intelligence, can produce content like text, images, audio, and synthetic data. It falls under deep learning, which uses artificial neural networks to process complex patterns. Generative AI differs from discriminative models by generating new data instances rather than classifying them. Transformers, a key advancement in natural language processing, consist of an encoder and decoder to handle input sequences and generate relevant outputs. Prompt design helps control model output, allowing users to generate content via browser-based prompts. Various generative AI models include text-to-text, text-to-image, text-to-video, and text-to-3D, each trained to generate specific outputs. Foundation models, pre-trained on vast data, can be adapted for tasks like sentiment analysis and image captioning. Tools like Google's Vertex AI, Gen AI Studio, App Builder, PaLM API, and Maker Suite facilitate creating and deploying generative AI models and applications, highlighting their broad potential across industries.",
    "The speaker discusses the question of how to speed up learning, particularly language learning, and shares insights from his lifelong exploration of this topic. He emphasizes the importance of focusing on relevant content, using language as a tool from day one, and understanding messages to acquire language unconsciously. He highlights five principles: attention, meaning, relevance, memory, and psychological state. He also outlines seven actions for language learning: listening a lot, getting the meaning before words, starting to mix words, focusing on core language, using language as a tool, having a language parent, and copying the face. The speaker dispels myths about talent and immersion, arguing that anyone can learn a new language to fluency in six months by following these principles and actions.",
    "The speaker, a former soccer coach, emphasizes the importance of self-confidence in achieving success. He recounts his interactions with parents and players, highlighting that self-confidence is a crucial skill for soccer players and beyond. He defines self-confidence as the belief in one's ability to accomplish tasks despite challenges and asserts that it can be developed through repetition and persistence. The speaker discusses the negative impact of self-talk and stresses the importance of positive self-affirmation. He shares his personal affirmation, 'I am the captain of my ship and the master of my fate,' and emphasizes the need to surround oneself with supportive people. He suggests building self-confidence in others by focusing on and praising positive behaviors. Lastly, he humorously illustrates how self-confident individuals interpret feedback in a way that reinforces their confidence. He concludes by encouraging belief in oneself, echoing the sentiments of the famous 'Here's to the crazy ones' speech.",
    "The speaker begins by asking how many people know the person next to them and recalls their first conversation. Conversations are likened to metal links, forming connections with every interaction. The speaker emphasizes the importance of talking to strangers, offering seven ways to start a conversation with anyone. Firstly, they stress the importance of breaking the ice with a simple 'Hi' to open the floodgates of conversation. Secondly, they recommend skipping small talk and asking personal questions to make the interaction meaningful. Thirdly, finding common ground or 'me too's helps build rapport. Fourthly, paying unique and genuine compliments makes people feel valued. Fifthly, asking for opinions opens up a two-way street of communication. Sixthly, being present and making eye contact ensures a genuine connection. Lastly, remembering details like names and places shows genuine interest and helps keep the conversation going. The speaker concludes by comparing conversations to reading books, encouraging the audience to delve deeper into people's stories rather than settling for superficial interactions.",
    "The speaker begins by sharing his parents' inspiring stories and the lessons they taught him about health and science. He discusses the global epidemic of unhealthy living, highlighting obesity and smoking as preventable causes of premature death. He introduces the concept of willingness, part of Acceptance and Commitment Therapy (ACT), as a new science of self-control. Willingness involves accepting cravings without acting on them, rather than trying to suppress them. The speaker's research at the Fred Hutchinson Cancer Research Center shows promising results in using willingness to help people quit smoking. He shares the story of 'Jane', a composite of his counseling experiences, who learned to be aware of her cravings and separate herself from her thoughts, reducing the power of those cravings. The key to self-control, he explains, is to give up control and drop the struggle with cravings, allowing them to pass. He encourages the audience to practice awareness and willingness with their own cravings and to treat themselves with kindness.",
    "Benjamin Todd discusses his journey to find a fulfilling career, despite varied interests in martial arts, philosophy, and finance. Realizing the lack of clear guidance, he founded '80000hours' to research optimal career choices. His findings challenge the advice of 'follow your passion,' which often leads to failure due to a mismatch between passions and job opportunities. Instead, Todd suggests focusing on doing what is valuable—pursuing careers that make a meaningful difference. He argues that mastering valuable skills and solving pressing social issues lead to personal fulfillment. Todd emphasizes exploring different fields, acquiring in-demand skills, and targeting neglected critical problems. He concludes that focusing on altruistic endeavors will naturally lead to a passionate and rewarding career, ensuring one's 80,000 working hours are not wasted.",
    "Tansel Ali recounts his struggles with traditional education and how he discovered a love for learning through experimentation. Initially disinterested in conventional learning methods, Ali found passion in studying psychology, leading to significant personal and academic achievements. He emphasizes the importance of effective memorization techniques, illustrating with a visualization exercise that helps recall the names of the past ten U.S. presidents. Ali argues that enhancing memory skills and embracing experimentation can lead to mastery in various fields. He proposes a three-step approach: checking current methods, experiencing new ones, and experimenting with these methods in real life. This strategy fosters continuous self-improvement, proving beneficial for all age groups. Ali concludes by advocating for teaching these techniques in schools and encouraging individuals to embrace lifelong learning.",
    "Kevin Bahler reflects on the evolution of his self-introductions, from childhood honesty about his interests to safer, socially acceptable introductions in high school and college. Initially hiding his true passions to fit in, he realized that genuine connections only formed when he embraced his authentic self. By sharing his interests openly, such as his love for chemistry, physics, and finger-painting, he found like-minded friends who enriched his life. Bahler challenges societal norms that define individuals by their jobs and advocates for expressing one's true passions, believing that genuine introductions can lead to meaningful relationships. He concludes with a personal introduction, emphasizing his love for seeing people truly happy.",
    "Social psychologist Amber Boydstun discusses how people think, particularly how our minds tend to get stuck on negatives. She explains that this negativity bias makes it easier to shift from good to bad rather than bad to good. To explore this, she conducted experiments where participants were given information framed either positively or negatively and then asked to reconsider it from the opposite perspective. The studies revealed that once people think about something negatively, they find it hard to see it positively again. This bias is evident in how people perceive the economy and their personal experiences. Boydstun suggests that to counteract this, individuals should actively practice focusing on positive aspects of life, such as writing about things they're grateful for and sharing good news with others. By retraining our minds, we can better appreciate the positives and improve our overall well-being.",
    "Art Benjamin, a 'mathemagician,' combines his love of math and magic to perform impressive mental calculations. During his presentation, he invites audience members with calculators to join him on stage to verify his calculations. He begins by squaring two-digit numbers faster than the audience members can with their calculators. He then moves on to three-digit numbers, quickly and accurately squaring them in his head. Benjamin further challenges himself by squaring a four-digit number and even a five-digit number, explaining his mental process and using mnemonic devices to help him remember intermediate results. Throughout his performance, he also demonstrates his ability to determine the day of the week for any given date, further showcasing his remarkable mental math skills. The audience is encouraged to interact, call out numbers, and verify his calculations, making the performance both engaging and impressive. Benjamin concludes with a complex five-digit squaring problem, thinking out loud to share his mental calculation process, and successfully arriving at the correct answer, earning applause and admiration from the audience.",
    "The speaker reveals a significant secret: school can make children less intelligent by stifling their creative intelligence despite increasing their academic knowledge. Sharing his own story, he recalls how at 14, he felt aimless until he discovered a passion for creating through a business plan competition. Winning that competition led to more victories and a realization of his love for innovation. However, when he and his friends attempted to recruit peers for their startup, they were mocked by older students but enthusiastically supported by younger kids. This experience highlighted how traditional education suppresses creativity over time, pushing students towards conventional paths of success rather than encouraging entrepreneurial thinking. The speaker urges parents and educators to inspire youth to explore diverse possibilities, innovate, and pursue entrepreneurship, emphasizing that true change comes from challenging societal norms and expectations.",
    "The speaker highlights a significant issue: only 26% of U.S. 12th graders are proficient in math. He argues that this isn't because most people aren't hardwired for math, but because math has been taught as a dehumanized subject. By treating math as a human language, akin to English or Spanish, it becomes more understandable. The speaker provides examples, like teaching fractions through relatable analogies, to show how math can be intuitive. He recounts how a frustrated high school student transformed her relationship with math by mastering multiplication, thereby gaining confidence and problem-solving skills. He emphasizes the urgency of adopting a language approach to math to improve proficiency and encourages educators to make math relatable and engaging. The ultimate goal is to raise the nation's math proficiency, fostering young minds to imagine and build a future.",
    "Jacob Barnett, diagnosed with autism at a young age, defied expectations by turning his unique way of thinking into an asset. He argues that true success comes from thinking creatively and not merely learning what is already known. Barnett highlights how historical figures like Isaac Newton and Albert Einstein made groundbreaking discoveries not through traditional learning but by thinking differently. He recounts his own journey from being unable to focus in special education to solving complex astrophysics problems and becoming a university student at age ten. Barnett emphasizes the importance of creativity over rote learning and challenges the audience to stop learning for 24 hours and instead think and create within their passions. He believes that everyone has the potential to innovate if they stop conforming to conventional educational paths and start thinking independently.",
    "In this talk, the speaker explores the intersection of hacking and technology, demonstrating how everyday devices like hotel TVs, car key fobs, door locks, and USB drives can be exploited for hacking. They emphasize the importance of viewing technology with a hacker's mindset to uncover vulnerabilities and potential security risks. The speaker shares various hacking projects, such as the 'Hackerbot' for Wi-Fi password sniffing and Bluetooth tracking at conferences. They also discuss the creation of a laser system to combat mosquitoes, aiming to reduce malaria transmission. The speaker highlights the value of curiosity and problem-solving skills in driving innovation and finding solutions to complex global challenges.",
    "Mark Rober conducted an experiment with his YouTube followers using a computer programming puzzle to explore how framing affects learning. The puzzle had two versions: one where failure resulted in no point loss and another where failure incurred a five-point penalty. The study found that those who didn't lose points had a higher success rate (68%) and made more attempts (two and a half times more) compared to those penalized for failure (52%). This demonstrated that framing failure in a less negative light encourages persistence and learning. Rober relates this to how toddlers learn to walk, focusing on successes rather than failures. He also likens it to playing video games, where the objective is to overcome challenges rather than dwell on failures. Rober shares his experience building a dartboard that guarantees bullseyes, highlighting that his motivation stemmed from the challenge and learning from setbacks, not fearing failure. He argues that reframing learning as a game can make it more engaging and less intimidating, thus enhancing success and knowledge acquisition. This 'Super Mario Effect' emphasizes focusing on the goal rather than the obstacles, promoting a mindset that turns challenges into opportunities for growth.",
    "Tim Doner, a teenage polyglot, shares his journey and passion for learning languages, which was featured in a New York Times article. Initially excited about the media attention, he soon became frustrated with how his story was sensationalized, focusing on the number of languages he spoke rather than the process and cultural insights gained. Doner emphasizes that knowing a language is more than just vocabulary; it's about understanding the culture and communication nuances. He discusses his methods for learning languages, such as using spatial memory and grouping similar-sounding words, and highlights the cultural richness and personal connections that come from truly engaging with a language. Doner concludes by stressing the importance of appreciating linguistic diversity and understanding that language represents a cultural worldview, which cannot be fully captured through simple translation.",
    "Mariya Khludnevskaya explores how the human brain can achieve native fluency in two languages from an early age, challenging the notion that bilingualism is inherently difficult or confusing. Through studies at the University of Washington's Institute for Learning & Brain Sciences, Khludnevskaya and her team found that bilingual babies' brains specialize in processing both languages they are exposed to, leading to cognitive benefits such as enhanced flexible thinking. Despite common concerns, bilingualism does not slow language learning or cause confusion; rather, code-switching is a sign of linguistic sophistication. She advocates for creating environments that foster bilingualism from a young age, emphasizing that interactive, socially rich settings are crucial for language acquisition. Khludnevskaya highlights promising research in Europe aimed at integrating bilingual education into public early education, which could revolutionize language learning and help children reach their full potential."
]

"""This loop prints each transcript, useful for verifying the transcripts fetched."""
for i, transcript in enumerate(transcripts):
    print(f"Transcript for Video {i + 1}:")
    print(transcript)
    print("\n" + "-"*80 + "\n")

"""
Creates a dictionary with transcripts and summaries and then creates a Hugging Face Dataset object from this dictionary.
"""
data = {'transcript': transcripts, 'summary': summaries}
dataset = Dataset.from_dict(data)

"""
Loads a pre-trained BART tokenizer and model from the specified directory.
"""
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


"""
The preprocess_function is designed to prepare the dataset for training a sequence-to-sequence model like BART. Here's a step-by-step breakdown:

Extract Transcripts: The function takes a batch of examples and extracts the 'transcript' field from each example, creating a list of input texts.

Tokenize Transcripts:It uses the BartTokenizer to convert the list of input texts into token IDs, ensuring that each sequence is no longer than 512 tokens. If a sequence is shorter, it will be padded to 512 tokens; if it is longer, it will be truncated.

Tokenize Summaries: Similarly, it converts the 'summary' field from each example into token IDs with the same maximum length, padding, and truncation settings.

Add Tokenized Summaries to Inputs:The tokenized summaries (labels) are added to the tokenized inputs under the 'labels' key. This structure is required by the Seq2SeqTrainer for training the model.

Return Processed Data:The function returns the processed dictionary, which includes the tokenized inputs and their corresponding tokenized summaries. This processed data will be used for training the model.
This preprocessing ensures that the inputs and targets are in the correct format and length, allowing the model to learn effectively from the provided data.
"""
def preprocess_function(examples):
    inputs = [doc for doc in examples['transcript']]
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)  # Ensure padding and truncation
    
    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples['summary'], max_length=512, padding='max_length', truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load metric
rouge = evaluate.load("rouge")


"""
The compute_metrics function is designed to evaluate the performance of the BART model by comparing its predicted summaries against the true summaries using the ROUGE metric. Here’s a step-by-step breakdown:

Unpack Predictions and Labels:The function receives a tuple containing the model's predictions and the true labels, which are unpacked into separate variables.

Decode Predictions:The predicted token IDs are converted into human-readable text using the tokenizer, and special tokens are skipped to get clean summaries.

Handle Special Tokens in Labels:The true labels contain -100 for ignored positions, which are replaced with the tokenizer's padding token ID to ensure they can be decoded correctly.

Decode Labels:The true token IDs are converted into human-readable text using the tokenizer, and special tokens are skipped to get clean summaries.

Compute ROUGE Scores:The decoded predicted summaries and true summaries are compared using the ROUGE metric, which measures the overlap of n-grams, word sequences, and word pairs between the predictions and references.

Adjust and Extract Results:The computed ROUGE scores are multiplied by 100 to convert them into percentages for easier interpretation.

Return Results:The function returns the adjusted ROUGE scores, which can be used to evaluate the model's performance.

By following these steps, the compute_metrics function provides a clear and quantifiable way to assess how well the BART model is generating summaries compared to the true summaries.
"""
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    return result

# Training arguments
"""
output_dir='./results'
Purpose: This specifies the directory where all the output files, including model checkpoints and evaluation results, will be saved.
Usage: It ensures that you have a designated place to store training artifacts, making it easier to organize and retrieve them later.

eval_strategy="epoch"
Purpose: This sets the evaluation strategy during training. By specifying "epoch", the model will be evaluated at the end of each epoch.
Usage: Regular evaluation helps in monitoring the model’s performance and making necessary adjustments during training.

learning_rate=2e-4
Purpose: This sets the learning rate for the optimizer. A slightly higher learning rate like 2e-4 can help in faster convergence but needs to be balanced to avoid overshooting.
Usage: The learning rate controls how much to change the model's parameters in response to the computed gradients during training. Finding the right learning rate is crucial for effective training.

per_device_train_batch_size=4
Purpose: This sets the batch size for training on each device (e.g., each GPU).
Usage: The batch size determines how many samples are processed before updating the model's parameters. A larger batch size can improve gradient estimation but requires more memory.

per_device_eval_batch_size=4
Purpose: This sets the batch size for evaluation on each device.
Usage: Similar to the training batch size, but for evaluation. It should be set considering the memory constraints and the need for consistent evaluation.

num_train_epochs=3
Purpose: This specifies the number of epochs to train the model.
Usage: An epoch is one complete pass through the entire training dataset. Training for more epochs can improve the model’s performance, but it may also lead to overfitting.

weight_decay=0.01
Purpose: This sets the weight decay (L2 regularization) rate.
Usage: Weight decay is used to prevent overfitting by penalizing large weights in the model, effectively regularizing the model.

logging_dir='./logs'
Purpose: This specifies the directory where the logs will be saved.
Usage: Logs are useful for tracking the training process, monitoring metrics, and debugging. The logging directory keeps these logs organized.

logging_steps=50
Purpose: This sets the interval (in steps) at which the training logs will be generated.
Usage: Frequent logging helps in monitoring the training process closely. However, logging too frequently can slow down training.

save_total_limit=1
Purpose: This specifies the maximum number of checkpoints to keep. If the limit is exceeded, older checkpoints are deleted.
Usage: Keeping only the most recent checkpoint saves disk space and ensures that you always have the latest model state.

save_steps=500
Purpose: This sets the interval (in steps) at which the model checkpoints will be saved.
Usage: Regular checkpointing ensures that you can resume training from the latest state in case of interruptions. It also provides intermediate models that can be evaluated or used for further fine-tuning.

predict_with_generate=True
Purpose: This flag indicates whether to use the models generate method to predict sequences during evaluation.
Usage: For sequence-to-sequence tasks like summarization, generation-based evaluation provides more meaningful metrics as it evaluates the actual generated text.
g
radient_accumulation_steps=2
Purpose: This sets the number of steps to accumulate gradients before performing a backward/update pass.
Usage: Gradient accumulation helps in effectively increasing the batch size without requiring additional memory. It simulates a larger batch size by accumulating gradients over multiple steps and updating the model parameters less frequently.

Summary
These parameters together control various aspects of the training process, from data handling and optimization to logging and checkpointing. Adjusting these parameters can significantly impact the efficiency, speed, and effectiveness of model training. Here's a quick summary of what each parameter does:
output_dir: Where to save training outputs.
eval_strategy: When to evaluate the model.
learning_rate: Controls how much the model updates during training.
per_device_train_batch_size: Number of training samples per batch per device.
per_device_eval_batch_size: Number of evaluation samples per batch per device.
num_train_epochs: Total passes through the training dataset.
weight_decay: Regularization to prevent overfitting.
logging_dir: Where to save log files.
logging_steps: How often to log training metrics.
save_total_limit: Maximum number of checkpoints to keep.
save_steps: How often to save model checkpoints.
predict_with_generate: Use generation for evaluation.
gradient_accumulation_steps: Accumulate gradients over multiple steps to simulate larger batch size.
By fine-tuning these parameters, you can optimize the training process to get the best possible performance from your model.
"""
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-4,  # Slightly increased learning rate
    per_device_train_batch_size=4,  # Increase batch size if GPU memory allows
    per_device_eval_batch_size=4,
    num_train_epochs=3,  # Increase to 3 epochs for better learning
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,  # Reduce frequency of logging
    save_total_limit=1,  # Keep only the most recent checkpoint
    save_steps=500,  # Save less frequently if needed
    predict_with_generate=True,
    gradient_accumulation_steps=2  # Accumulate gradients to simulate larger batch size
)

# Initialize Trainer
"""
The DataCollatorForSeq2Seq ensures that each batch of input data is correctly 
formatted and padded, which is essential for the efficient and correct training of 
sequence-to-sequence models like BART. By using the tokenizer and the model, 
it ensures compatibility and efficiency in data handling, facilitating a smooth training process.
"""
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


"""
Trainer
Model:The model is the core of the training process. For sequence-to-sequence tasks, it consists of an encoder to process the input sequence and a decoder to generate the output sequence. During training, the model's parameters are updated to minimize the loss, which measures the difference between the predicted and actual outputs.

Training Arguments:These arguments control the training loop, including how many epochs to train for, the learning rate, batch sizes, logging frequency, and checkpointing behavior. Adjusting these parameters can significantly impact the training efficiency and the final performance of the model.

Datasets:The training dataset provides the examples the model learns from. Each example consists of an input sequence (e.g., transcript) and a target sequence (e.g., summary). The evaluation dataset is used to periodically assess the model's performance and ensure it is learning correctly.

Tokenizer:The tokenizer converts the raw text into a format the model can process (token IDs) and back into human-readable text for evaluation and inference. It ensures consistency in text processing throughout the training pipeline.

Data Collator:The data collator prepares batches of data for training and evaluation. It handles padding, so all sequences in a batch are of the same length, and ensures the data is formatted correctly for the model.

Metrics Function: The metrics function evaluates the model's predictions against the ground truth. By computing scores like ROUGE, it provides a quantitative measure of the model's performance. This feedback is essential for understanding how well the model is performing and for making adjustments during training.
"""
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# Custom callback to generate summaries and compute ROUGE scores
"""The RougeCallback class provides a way to automatically generate 
summaries and compute ROUGE scores at the end of each training epoch. 
This custom callback helps in monitoring the model's performance and making 
necessary adjustments during training. By evaluating the generated summaries 
against reference summaries, it provides a clear and quantitative measure of 
the model's summarization quality.
"""
class RougeCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        for i in range(len(transcripts)):
            # Generate a summary for each transcript
            inputs = tokenizer(transcripts[i], return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=512, min_length=30, length_penalty=2.0, early_stopping=True)
            generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Compute ROUGE score
            reference_summary = summaries[i]
            result = rouge.compute(predictions=[generated_summary], references=[reference_summary], use_stemmer=True)
            result = {key: value * 100 for key, value in result.items()}
            
            print(f"\nEpoch {state.epoch}: Transcript {i + 1}")
            print(f"Generated Summary: {generated_summary}")
            print(f"Reference Summary: {reference_summary}")
            print(f"ROUGE Scores: {result}")

# Add the custom callback to the trainer
"""By adding this callback, you ensure that at the end of each epoch, 
summaries will be generated for the transcripts and ROUGE scores will be computed and printed. 
This provides ongoing feedback on the model’s performance during training."""
trainer.add_callback(RougeCallback())

# Train model
"""This method runs the training loop according to the specifications in training_args. 
It handles feeding batches of data to the model, computing the loss,
 updating the model parameters using backpropagation, and periodically evaluating 
 the model using the evaluation dataset. It also triggers any callbacks that have been added, 
 such as the RougeCallback."""
trainer.train()

# Save the model
"""Role: Saving the model and tokenizer allows you to reload them later for inference or further training. 
It ensures that the fine-tuning process is preserved and can be replicated or used in production."""
model.save_pretrained("./finetuned_bart_model")
tokenizer.save_pretrained("./finetuned_bart_model")

print("Model saved successfully.")
