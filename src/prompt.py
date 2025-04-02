from langchain_core.prompts import ChatPromptTemplate
prompts = ChatPromptTemplate.from_messages( [
    ("system",""" You are a helpfull assistant for question answering task.
    use the following retrived context to generate answer.Keep it short and concise.
    If you do not  find anything from the context then just say I do not know "
      "\n\n {context} """),

     ("human","{input}")
     ] )