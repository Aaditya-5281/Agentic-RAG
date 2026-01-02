from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool
# from tools.custom_tool import DocumentSearchTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool
from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool

# Global variables to store configuration
_pdf_path = None
_llm = None

def set_pdf_path(pdf_path: str):
    global _pdf_path
    _pdf_path = pdf_path

def set_llm(llm: LLM):
    global _llm
    _llm = llm

@CrewBase
class AgenticRag():
	"""AgenticRag crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def _get_pdf_tool(self):
		if _pdf_path:
			return DocumentSearchTool(file_path=_pdf_path)
		return None

	def _get_web_search_tool(self):
		return FireCrawlWebSearchTool()

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	# @agent
	# def routing_agent(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['routing_agent'],
	# 		verbose=True
	# 	)

	@agent
	def retriever_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['retriever_agent'],
			verbose=True,
			tools=[
				self._get_pdf_tool(),
				self._get_web_search_tool()
			],
			llm=_llm
		)

	@agent
	def response_synthesizer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['response_synthesizer_agent'],
			verbose=True,
			llm=_llm
		)

	# @task
	# def routing_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['routing_task'],
	# 	)

	@task
	def retrieval_task(self) -> Task:
		return Task(
			config=self.tasks_config['retrieval_task'],
		)

	@task
	def response_task(self) -> Task:
		return Task(
			config=self.tasks_config['response_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the AgenticRag crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
