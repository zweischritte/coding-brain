"""MCP Tools for code intelligence.

This module provides MCP tools for code intelligence:
- explain_code: Explain a code symbol with call graph and context
- search_code_hybrid: Tri-hybrid code search (lexical + semantic + graph)
- find_callers: Find functions that call a symbol
- find_callees: Find functions called by a symbol
- get_symbol_definition: Get symbol definition and location
- impact_analysis: Analyze impact of changes
- adr_automation: Automated ADR detection and generation (FR-014)
- test_generation: Automated test generation (FR-008)
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    # explain_code tool
    "ExplainCodeConfig",
    "ExplainCodeTool",
    "SymbolExplanation",
    "SymbolLookup",
    "CallGraphTraverser",
    "DocumentationExtractor",
    "CodeContextRetriever",
    "ExplanationFormatter",
    "ExplainCodeError",
    "SymbolNotFoundError",
    "InvalidSymbolIDError",
    "SymbolLookupError",
    "create_explain_code_tool",
    # search_code_hybrid tool
    "SearchCodeHybridConfig",
    "SearchCodeHybridInput",
    "SearchCodeHybridResult",
    "SearchCodeHybridTool",
    "SearchCodeHybridError",
    "InvalidQueryError",
    "CodeHit",
    "CodeSymbol",
    "create_search_code_hybrid_tool",
    # call_graph tools (find_callers, find_callees)
    "CallGraphConfig",
    "CallGraphInput",
    "GraphOutput",
    "GraphNode",
    "GraphEdge",
    "FindCallersTool",
    "FindCalleesTool",
    "CallGraphError",
    "create_find_callers_tool",
    "create_find_callees_tool",
    # symbol_definition tool
    "SymbolDefinitionConfig",
    "SymbolLookupInput",
    "SymbolDefinitionOutput",
    "GetSymbolDefinitionTool",
    "SymbolDefinitionError",
    "create_get_symbol_definition_tool",
    # impact_analysis tool
    "ImpactAnalysisConfig",
    "ImpactInput",
    "ImpactOutput",
    "AffectedFile",
    "ImpactAnalysisTool",
    "ImpactAnalysisError",
    "create_impact_analysis_tool",
    # adr_automation tool (FR-014)
    "ADRConfig",
    "ADRHeuristic",
    "DetectionResult",
    "AggregateResult",
    "DependencyHeuristic",
    "APIChangeHeuristic",
    "ConfigurationHeuristic",
    "SchemaHeuristic",
    "SecurityHeuristic",
    "PatternHeuristic",
    "CrossCuttingHeuristic",
    "PerformanceHeuristic",
    "ADRHeuristicEngine",
    "ChangeAnalyzer",
    "ADRTemplate",
    "ADRContext",
    "ADRGenerator",
    "AnalysisResult",
    "ADRAutomationTool",
    "ADRAutomationError",
    "HeuristicError",
    "TemplateError",
    "create_adr_automation_tool",
    # test_generation tool (FR-008)
    "TestGenerationConfig",
    "SymbolAnalyzer",
    "TestPattern",
    "PatternMatcher",
    "CoverageAnalyzer",
    "TestTemplate",
    "TestCase",
    "TestSuite",
    "TestGenerator",
    "TestGenerationTool",
    "TestGenerationError",
    "create_test_generation_tool",
]

# Module mappings for lazy imports
_MODULE_MAP = {
    # explain_code
    "ExplainCodeConfig": "explain_code",
    "ExplainCodeTool": "explain_code",
    "SymbolExplanation": "explain_code",
    "SymbolLookup": "explain_code",
    "CallGraphTraverser": "explain_code",
    "DocumentationExtractor": "explain_code",
    "CodeContextRetriever": "explain_code",
    "ExplanationFormatter": "explain_code",
    "ExplainCodeError": "explain_code",
    "SymbolNotFoundError": "explain_code",
    "InvalidSymbolIDError": "explain_code",
    "SymbolLookupError": "explain_code",
    "create_explain_code_tool": "explain_code",
    # search_code_hybrid
    "SearchCodeHybridConfig": "search_code_hybrid",
    "SearchCodeHybridInput": "search_code_hybrid",
    "SearchCodeHybridResult": "search_code_hybrid",
    "SearchCodeHybridTool": "search_code_hybrid",
    "SearchCodeHybridError": "search_code_hybrid",
    "InvalidQueryError": "search_code_hybrid",
    "CodeHit": "search_code_hybrid",
    "CodeSymbol": "search_code_hybrid",
    "create_search_code_hybrid_tool": "search_code_hybrid",
    # call_graph
    "CallGraphConfig": "call_graph",
    "CallGraphInput": "call_graph",
    "GraphOutput": "call_graph",
    "GraphNode": "call_graph",
    "GraphEdge": "call_graph",
    "FindCallersTool": "call_graph",
    "FindCalleesTool": "call_graph",
    "CallGraphError": "call_graph",
    "create_find_callers_tool": "call_graph",
    "create_find_callees_tool": "call_graph",
    # symbol_definition
    "SymbolDefinitionConfig": "symbol_definition",
    "SymbolLookupInput": "symbol_definition",
    "SymbolDefinitionOutput": "symbol_definition",
    "GetSymbolDefinitionTool": "symbol_definition",
    "SymbolDefinitionError": "symbol_definition",
    "create_get_symbol_definition_tool": "symbol_definition",
    # impact_analysis
    "ImpactAnalysisConfig": "impact_analysis",
    "ImpactInput": "impact_analysis",
    "ImpactOutput": "impact_analysis",
    "AffectedFile": "impact_analysis",
    "ImpactAnalysisTool": "impact_analysis",
    "ImpactAnalysisError": "impact_analysis",
    "create_impact_analysis_tool": "impact_analysis",
    # adr_automation
    "ADRConfig": "adr_automation",
    "ADRHeuristic": "adr_automation",
    "DetectionResult": "adr_automation",
    "AggregateResult": "adr_automation",
    "DependencyHeuristic": "adr_automation",
    "APIChangeHeuristic": "adr_automation",
    "ConfigurationHeuristic": "adr_automation",
    "SchemaHeuristic": "adr_automation",
    "SecurityHeuristic": "adr_automation",
    "PatternHeuristic": "adr_automation",
    "CrossCuttingHeuristic": "adr_automation",
    "PerformanceHeuristic": "adr_automation",
    "ADRHeuristicEngine": "adr_automation",
    "ChangeAnalyzer": "adr_automation",
    "ADRTemplate": "adr_automation",
    "ADRContext": "adr_automation",
    "ADRGenerator": "adr_automation",
    "AnalysisResult": "adr_automation",
    "ADRAutomationTool": "adr_automation",
    "ADRAutomationError": "adr_automation",
    "HeuristicError": "adr_automation",
    "TemplateError": "adr_automation",
    "create_adr_automation_tool": "adr_automation",
    # test_generation
    "TestGenerationConfig": "test_generation",
    "SymbolAnalyzer": "test_generation",
    "TestPattern": "test_generation",
    "PatternMatcher": "test_generation",
    "CoverageAnalyzer": "test_generation",
    "TestTemplate": "test_generation",
    "TestCase": "test_generation",
    "TestSuite": "test_generation",
    "TestGenerator": "test_generation",
    "TestGenerationTool": "test_generation",
    "TestGenerationError": "test_generation",
    "create_test_generation_tool": "test_generation",
}


def __getattr__(name):
    """Lazy import attributes."""
    if name in _MODULE_MAP:
        import importlib

        module_name = _MODULE_MAP[name]
        module = importlib.import_module(f"openmemory.api.tools.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
