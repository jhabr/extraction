import crewai


class ExtractorAgent(crewai.Agent):
    def __init__(self):
        super().__init__(
            role="Document Information Extractor",
            goal="Extract all information from a given document.",
            backstory=(
                "You are a expert at information extraction form documents."
                "Your expertise lies in the correct extraction of all information from a given document."
            ),
            verbose=True,
            allow_delegation=True,
            max_rpm=10,
        )


class ReviewerAgent(crewai.Agent):
    def __init__(self):
        super().__init__(
            role="Document Information Reviewer",
            goal=(
                "Review all extracted information from a given document and make sure the extracted data "
                "is a 100% correct."
            ),
            backstory=(
                "You are a thorough reviewer of information that has been extracted from a document."
                "Your expertise lies in uncovering errors in the extraction and providing them as a feedback."
            ),
            verbose=True,
            allow_delegation=False,
        )
