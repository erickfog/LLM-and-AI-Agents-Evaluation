"""
Métricas específicas de avaliação para Agents usando Gemini como LLM-as-a-Judge.

As funções abaixo constroem prompts de avaliação e usam o Gemini para:
- Task Completion
- Tool Correctness
- Argument Correctness
- Plan Quality
- Plan Adherence
- Step Efficiency
"""

from typing import List, Dict, Optional, Any
import json
import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


def _get_gemini_model_name() -> str:
    """
    Retorna o nome do modelo Gemini a ser usado como juiz.
    """
    return (
        os.getenv("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL_NAME")
        or "gemini-1.5-flash"
    )


def _configure_gemini() -> None:
    """
    Configura o cliente do Gemini usando GEMINI_API_KEY ou GOOGLE_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Configure GEMINI_API_KEY (ou GOOGLE_API_KEY) no seu ambiente/.env "
            "para usar o Gemini como modelo juiz."
        )
    genai.configure(api_key=api_key)


class GeminiJudgeModel:
    """
    Wrapper simples sobre google.generativeai para expor um método `.generate(prompt)`.
    """

    def __init__(self) -> None:
        _configure_gemini()
        self._model = genai.GenerativeModel(_get_gemini_model_name())

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text
        try:
            return "".join(
                getattr(part, "text", "")
                for candidate in getattr(response, "candidates", [])  # type: ignore[assignment]
                for part in getattr(candidate, "content", []).parts
            ).strip()
        except Exception:
            return str(response)


class AgentAction:
    """Representa uma ação executada por um agente"""
    def __init__(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        step_number: int
    ):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.step_number = step_number
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'arguments': self.arguments,
            'result': str(self.result),
            'step_number': self.step_number
        }


class AgentExecution:
    """Representa uma execução completa de um agente"""
    def __init__(
        self,
        task: str,
        plan: Optional[List[str]] = None,
        actions: Optional[List[AgentAction]] = None,
        final_output: Optional[str] = None
    ):
        self.task = task
        self.plan = plan or []
        self.actions = actions or []
        self.final_output = final_output
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task,
            'plan': self.plan,
            'actions': [action.to_dict() for action in self.actions],
            'final_output': self.final_output
        }


def evaluate_task_completion(
    agent_execution: AgentExecution,
    expected_output: Optional[str] = None,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Avalia se o agente completou a tarefa com sucesso.
    
    Args:
        agent_execution: Objeto AgentExecution com a execução do agente
        expected_output: Output esperado (opcional)
        threshold: Threshold mínimo para considerar sucesso (0-1)
    
    Returns:
        Dict com score, success, reason e outros metadados
    """
    evaluation_model = GeminiJudgeModel()
    
    task = agent_execution.task
    final_output = agent_execution.final_output or "Nenhuma saída final"
    actions_summary = f"{len(agent_execution.actions)} ações executadas"
    
    prompt = f"""Avalie se o agente completou a tarefa com sucesso.

Tarefa: {task}
Saída Final: {final_output}
Ações Executadas: {actions_summary}

"""
    
    if expected_output:
        prompt += f"Saída Esperada: {expected_output}\n\n"
    
    prompt += """Responda no formato:
SCORE: [número de 0 a 1, onde 1.0 = tarefa completamente concluída]
REASONING: [explicação detalhada]"""
    
    try:
        response = evaluation_model.generate(prompt)
        
        # Extrair score
        lines = response.split('\n')
        score = None
        reasoning = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                score_str = line.split('SCORE:')[1].strip()
                score = float(score_str)
            elif line.startswith('REASONING:'):
                reasoning.append(line.split('REASONING:')[1].strip())
            elif score is None:
                try:
                    score = float(line.strip())
                except:
                    pass
            else:
                reasoning.append(line.strip())
        
        if score is None:
            score = 0.5
        
        success = score >= threshold
        reason = ' '.join(reasoning) if reasoning else response
        
        return {
            'score': score,
            'success': success,
            'reason': reason,
            'metric_name': 'Task Completion',
            'task': task,
            'final_output': final_output
        }
    except Exception as e:
        return {
            'score': 0.0,
            'success': False,
            'reason': f'Erro na avaliação: {str(e)}',
            'metric_name': 'Task Completion',
            'error': str(e)
        }


def evaluate_tool_correctness(
    agent_execution: AgentExecution,
    expected_tools: Optional[List[str]] = None,
    threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Avalia se as ferramentas corretas foram chamadas.
    
    Args:
        agent_execution: Objeto AgentExecution com a execução do agente
        expected_tools: Lista de ferramentas esperadas (opcional)
        threshold: Threshold mínimo para considerar sucesso (0-1)
    
    Returns:
        Dict com score, success, reason e outros metadados
    """
    evaluation_model = GeminiJudgeModel()
    
    task = agent_execution.task
    used_tools = [action.tool_name for action in agent_execution.actions]
    
    prompt = f"""Avalie se o agente usou as ferramentas corretas para completar a tarefa.

Tarefa: {task}
Ferramentas Usadas: {', '.join(used_tools) if used_tools else 'Nenhuma'}

"""
    
    if expected_tools:
        prompt += f"Ferramentas Esperadas: {', '.join(expected_tools)}\n\n"
    
    prompt += """Responda no formato:
SCORE: [número de 0 a 1, onde 1.0 = ferramentas perfeitamente adequadas]
REASONING: [explicação detalhada]"""
    
    try:
        response = evaluation_model.generate(prompt)
        
        # Extrair score
        lines = response.split('\n')
        score = None
        reasoning = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                score_str = line.split('SCORE:')[1].strip()
                score = float(score_str)
            elif line.startswith('REASONING:'):
                reasoning.append(line.split('REASONING:')[1].strip())
            elif score is None:
                try:
                    score = float(line.strip())
                except:
                    pass
            else:
                reasoning.append(line.strip())
        
        if score is None:
            # Fallback: verificação simples se expected_tools fornecido
            if expected_tools:
                matches = sum(1 for tool in used_tools if tool in expected_tools)
                score = matches / len(expected_tools) if expected_tools else 0.5
            else:
                score = 0.5
        
        success = score >= threshold
        reason = ' '.join(reasoning) if reasoning else response
        
        return {
            'score': score,
            'success': success,
            'reason': reason,
            'metric_name': 'Tool Correctness',
            'used_tools': used_tools,
            'expected_tools': expected_tools
        }
    except Exception as e:
        return {
            'score': 0.0,
            'success': False,
            'reason': f'Erro na avaliação: {str(e)}',
            'metric_name': 'Tool Correctness',
            'error': str(e)
        }


def evaluate_argument_correctness(
    agent_execution: AgentExecution,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Avalia se os argumentos passados para as ferramentas estão corretos.
    
    Args:
        agent_execution: Objeto AgentExecution com a execução do agente
        threshold: Threshold mínimo para considerar sucesso (0-1)
    
    Returns:
        Dict com score, success, reason e outros metadados
    """
    evaluation_model = GeminiJudgeModel()
    
    task = agent_execution.task
    actions_details = []
    
    for action in agent_execution.actions:
        actions_details.append(
            f"Ferramenta: {action.tool_name}, "
            f"Argumentos: {json.dumps(action.arguments, ensure_ascii=False)}"
        )
    
    prompt = f"""Avalie se os argumentos passados para cada ferramenta estão corretos e adequados.

Tarefa: {task}

Ações Executadas:
{chr(10).join(actions_details) if actions_details else 'Nenhuma ação'}

Responda no formato:
SCORE: [número de 0 a 1, onde 1.0 = todos os argumentos corretos]
REASONING: [explicação detalhada, mencionando quais argumentos estão corretos/incorretos]"""
    
    try:
        response = evaluation_model.generate(prompt)
        
        # Extrair score
        lines = response.split('\n')
        score = None
        reasoning = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                score_str = line.split('SCORE:')[1].strip()
                score = float(score_str)
            elif line.startswith('REASONING:'):
                reasoning.append(line.split('REASONING:')[1].strip())
            elif score is None:
                try:
                    score = float(line.strip())
                except:
                    pass
            else:
                reasoning.append(line.strip())
        
        if score is None:
            score = 0.5
        
        success = score >= threshold
        reason = ' '.join(reasoning) if reasoning else response
        
        return {
            'score': score,
            'success': success,
            'reason': reason,
            'metric_name': 'Argument Correctness',
            'num_actions': len(agent_execution.actions)
        }
    except Exception as e:
        return {
            'score': 0.0,
            'success': False,
            'reason': f'Erro na avaliação: {str(e)}',
            'metric_name': 'Argument Correctness',
            'error': str(e)
        }


def evaluate_plan_quality(
    agent_execution: AgentExecution,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Avalia a qualidade do plano de ação do agente.
    
    Args:
        agent_execution: Objeto AgentExecution com a execução do agente
        threshold: Threshold mínimo para considerar sucesso (0-1)
    
    Returns:
        Dict com score, success, reason e outros metadados
    """
    evaluation_model = GeminiJudgeModel()
    
    task = agent_execution.task
    plan = agent_execution.plan
    
    prompt = f"""Avalie a qualidade do plano de ação do agente.

Tarefa: {task}
Plano: {chr(10).join(f'{i+1}. {step}' for i, step in enumerate(plan)) if plan else 'Nenhum plano explícito'}

Critérios de avaliação:
- O plano é lógico e coerente?
- O plano cobre todos os aspectos necessários da tarefa?
- Os passos estão na ordem correta?
- O plano é eficiente (sem passos desnecessários)?

Responda no formato:
SCORE: [número de 0 a 1, onde 1.0 = plano perfeito]
REASONING: [explicação detalhada]"""
    
    try:
        response = evaluation_model.generate(prompt)
        
        # Extrair score
        lines = response.split('\n')
        score = None
        reasoning = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                score_str = line.split('SCORE:')[1].strip()
                score = float(score_str)
            elif line.startswith('REASONING:'):
                reasoning.append(line.split('REASONING:')[1].strip())
            elif score is None:
                try:
                    score = float(line.strip())
                except:
                    pass
            else:
                reasoning.append(line.strip())
        
        if score is None:
            score = 0.5
        
        success = score >= threshold
        reason = ' '.join(reasoning) if reasoning else response
        
        return {
            'score': score,
            'success': success,
            'reason': reason,
            'metric_name': 'Plan Quality',
            'plan': plan
        }
    except Exception as e:
        return {
            'score': 0.0,
            'success': False,
            'reason': f'Erro na avaliação: {str(e)}',
            'metric_name': 'Plan Quality',
            'error': str(e)
        }


def evaluate_plan_adherence(
    agent_execution: AgentExecution,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Avalia se o agente seguiu seu próprio plano.
    
    Args:
        agent_execution: Objeto AgentExecution com a execução do agente
        threshold: Threshold mínimo para considerar sucesso (0-1)
    
    Returns:
        Dict com score, success, reason e outros metadados
    """
    evaluation_model = GeminiJudgeModel()
    
    task = agent_execution.task
    plan = agent_execution.plan
    actions = agent_execution.actions
    
    plan_str = chr(10).join(f'{i+1}. {step}' for i, step in enumerate(plan)) if plan else 'Nenhum plano'
    actions_str = chr(10).join(
        f'{i+1}. {action.tool_name} com {json.dumps(action.arguments, ensure_ascii=False)}'
        for i, action in enumerate(actions)
    ) if actions else 'Nenhuma ação'
    
    prompt = f"""Avalie se o agente seguiu o plano que ele mesmo criou.

Tarefa: {task}

Plano Original:
{plan_str}

Ações Realizadas:
{actions_str}

Responda no formato:
SCORE: [número de 0 a 1, onde 1.0 = seguiu o plano perfeitamente]
REASONING: [explicação detalhada, mencionando desvios do plano se houver]"""
    
    try:
        response = evaluation_model.generate(prompt)
        
        # Extrair score
        lines = response.split('\n')
        score = None
        reasoning = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                score_str = line.split('SCORE:')[1].strip()
                score = float(score_str)
            elif line.startswith('REASONING:'):
                reasoning.append(line.split('REASONING:')[1].strip())
            elif score is None:
                try:
                    score = float(line.strip())
                except:
                    pass
            else:
                reasoning.append(line.strip())
        
        if score is None:
            score = 0.5
        
        success = score >= threshold
        reason = ' '.join(reasoning) if reasoning else response
        
        return {
            'score': score,
            'success': success,
            'reason': reason,
            'metric_name': 'Plan Adherence',
            'plan_steps': len(plan) if plan else 0,
            'actions_taken': len(actions)
        }
    except Exception as e:
        return {
            'score': 0.0,
            'success': False,
            'reason': f'Erro na avaliação: {str(e)}',
            'metric_name': 'Plan Adherence',
            'error': str(e)
        }


def evaluate_step_efficiency(
    agent_execution: AgentExecution,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Avalia a eficiência das ações do agente (número de passos, redundâncias, etc.).
    
    Args:
        agent_execution: Objeto AgentExecution com a execução do agente
        threshold: Threshold mínimo para considerar sucesso (0-1)
    
    Returns:
        Dict com score, success, reason e outros metadados
    """
    evaluation_model = GeminiJudgeModel()
    
    task = agent_execution.task
    num_actions = len(agent_execution.actions)
    actions_summary = f"{num_actions} ações executadas"
    
    # Detectar possíveis redundâncias
    tool_counts = {}
    for action in agent_execution.actions:
        tool_counts[action.tool_name] = tool_counts.get(action.tool_name, 0) + 1
    
    prompt = f"""Avalie a eficiência das ações do agente.

Tarefa: {task}
Número de Ações: {num_actions}
Ferramentas Usadas: {json.dumps(tool_counts, ensure_ascii=False)}

Critérios:
- O número de passos é razoável para a complexidade da tarefa?
- Há ações redundantes ou desnecessárias?
- As ações são diretas e objetivas?

Responda no formato:
SCORE: [número de 0 a 1, onde 1.0 = altamente eficiente]
REASONING: [explicação detalhada]"""
    
    try:
        response = evaluation_model.generate(prompt)
        
        # Extrair score
        lines = response.split('\n')
        score = None
        reasoning = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                score_str = line.split('SCORE:')[1].strip()
                score = float(score_str)
            elif line.startswith('REASONING:'):
                reasoning.append(line.split('REASONING:')[1].strip())
            elif score is None:
                try:
                    score = float(line.strip())
                except:
                    pass
            else:
                reasoning.append(line.strip())
        
        if score is None:
            # Fallback: score baseado em número de ações (menos ações = mais eficiente)
            # Normalizado para 0-1 (assumindo que <10 ações é bom)
            score = max(0, 1 - (num_actions / 20))
        
        success = score >= threshold
        reason = ' '.join(reasoning) if reasoning else response
        
        return {
            'score': score,
            'success': success,
            'reason': reason,
            'metric_name': 'Step Efficiency',
            'num_actions': num_actions,
            'tool_usage': tool_counts
        }
    except Exception as e:
        return {
            'score': 0.0,
            'success': False,
            'reason': f'Erro na avaliação: {str(e)}',
            'metric_name': 'Step Efficiency',
            'error': str(e)
        }


def evaluate_agent_batch(
    agent_executions: List[AgentExecution],
    metrics: List[str] = ['task_completion', 'tool_correctness'],
    thresholds: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Avalia múltiplas execuções de agentes com múltiplas métricas.
    
    Args:
        agent_executions: Lista de objetos AgentExecution
        metrics: Lista de métricas a aplicar
        thresholds: Dict com thresholds customizados por métrica
    
    Returns:
        Lista de resultados, um dict por execução
    """
    if thresholds is None:
        thresholds = {
            'task_completion': 0.7,
            'tool_correctness': 0.8,
            'argument_correctness': 0.7,
            'plan_quality': 0.7,
            'plan_adherence': 0.7,
            'step_efficiency': 0.7
        }
    
    results = []
    
    for i, execution in enumerate(agent_executions):
        case_results = {
            'execution_id': i,
            'task': execution.task,
            'metrics': {}
        }
        
        # Task Completion
        if 'task_completion' in metrics:
            result = evaluate_task_completion(
                agent_execution=execution,
                threshold=thresholds.get('task_completion', 0.7)
            )
            case_results['metrics']['task_completion'] = result
        
        # Tool Correctness
        if 'tool_correctness' in metrics:
            result = evaluate_tool_correctness(
                agent_execution=execution,
                threshold=thresholds.get('tool_correctness', 0.8)
            )
            case_results['metrics']['tool_correctness'] = result
        
        # Argument Correctness
        if 'argument_correctness' in metrics:
            result = evaluate_argument_correctness(
                agent_execution=execution,
                threshold=thresholds.get('argument_correctness', 0.7)
            )
            case_results['metrics']['argument_correctness'] = result
        
        # Plan Quality
        if 'plan_quality' in metrics:
            result = evaluate_plan_quality(
                agent_execution=execution,
                threshold=thresholds.get('plan_quality', 0.7)
            )
            case_results['metrics']['plan_quality'] = result
        
        # Plan Adherence
        if 'plan_adherence' in metrics:
            result = evaluate_plan_adherence(
                agent_execution=execution,
                threshold=thresholds.get('plan_adherence', 0.7)
            )
            case_results['metrics']['plan_adherence'] = result
        
        # Step Efficiency
        if 'step_efficiency' in metrics:
            result = evaluate_step_efficiency(
                agent_execution=execution,
                threshold=thresholds.get('step_efficiency', 0.7)
            )
            case_results['metrics']['step_efficiency'] = result
        
        # Calcular score médio
        scores = [m.get('score', 0) for m in case_results['metrics'].values()]
        case_results['average_score'] = sum(scores) / len(scores) if scores else 0.0
        case_results['all_passed'] = all(m.get('success', False) for m in case_results['metrics'].values())
        
        results.append(case_results)
    
    return results
