"""
core/tool_registry.py
─────────────────────
Central registry of every tool in the system.
Adding a new tool = one import + one line here.
"""

from tools.rag.embed_query      import EmbedQueryTool
from tools.rag.retrieve_docs    import RetrieveDocsTool
from tools.rag.grade_relevance  import GradeRelevanceTool
from tools.rag.generate_answer  import GenerateAnswerTool
from tools.rag.rewrite_query    import RewriteQueryTool

from tools.dfmea.case_router        import CaseRouterTool
from tools.dfmea.rag_context        import DFMEARagContextTool
from tools.dfmea.parse_import       import ParseImportTool
from tools.dfmea.generate_elements  import GenerateElementsTool
from tools.dfmea.generate_functions import GenerateFunctionsTool
from tools.dfmea.generate_failures  import GenerateFailuresTool
from tools.dfmea.generate_causes    import GenerateCausesTool
from tools.dfmea.rate_risks         import RateRisksTool
from tools.dfmea.assemble_output    import AssembleOutputTool
from tools.dfmea.export_xlsx        import ExportXlsxTool

from tools.optimizer.initialize       import InitializeOptTool
from tools.optimizer.solve_ode        import SolveOdeTool
from tools.optimizer.compute_rms      import ComputeRmsTool
from tools.optimizer.propose_k        import ProposeKTool
from tools.optimizer.check_convergence import CheckConvergenceTool
from tools.optimizer.summarize        import SummarizeOptTool

TOOL_REGISTRY: dict = {
    # RAG
    "embed_query":      EmbedQueryTool(),
    "retrieve_docs":    RetrieveDocsTool(),
    "grade_relevance":  GradeRelevanceTool(),
    "generate_answer":  GenerateAnswerTool(),
    "rewrite_query":    RewriteQueryTool(),

    # DFMEA
    "case_router":          CaseRouterTool(),
    "dfmea_rag_context":    DFMEARagContextTool(),
    "parse_import":         ParseImportTool(),
    "generate_elements":    GenerateElementsTool(),
    "generate_functions":   GenerateFunctionsTool(),
    "generate_failures":    GenerateFailuresTool(),
    "generate_causes":      GenerateCausesTool(),
    "rate_risks":           RateRisksTool(),
    "assemble_output":      AssembleOutputTool(),
    "export_xlsx":          ExportXlsxTool(),

    # Optimizer
    "init_opt":             InitializeOptTool(),
    "solve_ode":            SolveOdeTool(),
    "compute_rms":          ComputeRmsTool(),
    "propose_k":            ProposeKTool(),
    "check_convergence":    CheckConvergenceTool(),
    "summarize_opt":        SummarizeOptTool(),
}
