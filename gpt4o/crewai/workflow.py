import crewai

from gpt4o.crewai.agents import ExtractorAgent, ReviewerAgent


class ExtractionReviewWorkflow(crewai.Workflow):
    def __init__(self):
        super().__init__(
            name="Image Extraction and Review Workflow",
            description="A workflow where the ExtractorAgent extracts data and the ReviewerAgent validates it.",
        )

        # Define crewai in the workflow
        self.extractor_agent = ExtractorAgent()
        self.reviewer_agent = ReviewerAgent()

    def run(self, image_data):
        # Step 1: Extract the information using the ExtractorAgent
        extracted_data = self.extractor_agent.task(image_data)

        # Step 2: Review the extracted information using the ReviewerAgent
        reviewed_data = self.reviewer_agent.task(extracted_data)

        return reviewed_data
