"""
Exemplo completo de avaliação de Agent usando múltiplas métricas
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics_agents import (
    AgentExecution,
    AgentAction,
    evaluate_agent_batch
)
from src.utils import load_agent_dataset, format_results, generate_report, plot_metrics


def create_agent_execution_from_dict(data: dict) -> AgentExecution:
    """
    Cria um objeto AgentExecution a partir de um dict do dataset.
    """
    execution_data = data.get('execution', {})
    
    # Criar ações
    actions = []
    for action_data in execution_data.get('actions', []):
        action = AgentAction(
            tool_name=action_data['tool_name'],
            arguments=action_data['arguments'],
            result=action_data['result'],
            step_number=action_data['step_number']
        )
        actions.append(action)
    
    # Criar execução
    execution = AgentExecution(
        task=data['task'],
        plan=execution_data.get('plan', []),
        actions=actions,
        final_output=execution_data.get('final_output')
    )
    
    return execution


def main():
    """Executa avaliação completa de Agent"""
    
    print("=" * 80)
    print("EXEMPLO: AVALIAÇÃO DE AGENT COM MÚLTIPLAS MÉTRICAS")
    print("=" * 80)
    print()
    
    # 1. Carregar dataset
    print("1. Carregando dataset de teste...")
    dataset_path = Path(__file__).parent.parent / "data" / "test_dataset_agents.json"
    
    try:
        test_cases = load_agent_dataset(str(dataset_path))
        print(f"   ✓ Carregados {len(test_cases)} casos de teste")
    except Exception as e:
        print(f"   ✗ Erro ao carregar dataset: {e}")
        return
    
    print()
    
    # 2. Converter para objetos AgentExecution
    print("2. Preparando execuções de agentes...")
    agent_executions = []
    
    for case in test_cases:
        try:
            execution = create_agent_execution_from_dict(case)
            agent_executions.append(execution)
        except Exception as e:
            print(f"   ⚠ Erro ao processar caso {case.get('id', 'unknown')}: {e}")
            continue
    
    print(f"   ✓ {len(agent_executions)} execuções preparadas")
    print()
    
    # 3. Definir métricas a avaliar
    print("3. Configurando métricas...")
    metrics = [
        'task_completion',
        'tool_correctness',
        'argument_correctness',
        'plan_quality',
        'plan_adherence',
        'step_efficiency'
    ]
    
    print(f"   ✓ Métricas selecionadas: {', '.join(metrics)}")
    print()
    
    # 4. Executar avaliações
    print("4. Executando avaliações (isso pode levar alguns minutos)...")
    print("   Aguarde enquanto as métricas são calculadas...")
    print()
    
    try:
        results = evaluate_agent_batch(
            agent_executions=agent_executions,
            metrics=metrics,
            thresholds={
                'task_completion': 0.7,
                'tool_correctness': 0.8,
                'argument_correctness': 0.7,
                'plan_quality': 0.7,
                'plan_adherence': 0.7,
                'step_efficiency': 0.7
            }
        )
        print(f"   ✓ Avaliação concluída para {len(results)} execuções")
    except Exception as e:
        print(f"   ✗ Erro durante avaliação: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 5. Exibir resultados resumidos
    print("5. Resultados Resumidos:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        exec_id = result.get('execution_id', i)
        task = result.get('task', 'N/A')
        avg_score = result.get('average_score', 0.0)
        all_passed = result.get('all_passed', False)
        
        print(f"\nExecução #{exec_id}:")
        print(f"  Tarefa: {task[:60]}...")
        print(f"  Score Médio: {avg_score:.3f}")
        print(f"  Status: {'✓ PASSOU' if all_passed else '✗ FALHOU'}")
        
        if 'metrics' in result:
            for metric_name, metric_result in result['metrics'].items():
                score = metric_result.get('score', 0.0)
                success = metric_result.get('success', False)
                status = "✓" if success else "✗"
                print(f"    {status} {metric_name}: {score:.3f}")
    
    print()
    
    # 6. Gerar relatório
    print("6. Gerando relatório...")
    report_path = Path(__file__).parent.parent / "reports" / "agent_evaluation_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    report = generate_report(results, output_file=str(report_path), include_details=True)
    print(f"   ✓ Relatório salvo em: {report_path}")
    print()
    
    # 7. Gerar visualizações
    print("7. Gerando visualizações...")
    plot_path = Path(__file__).parent.parent / "reports" / "agent_evaluation_plots.png"
    
    try:
        plot_metrics(results, output_file=str(plot_path))
        print(f"   ✓ Gráficos salvos em: {plot_path}")
    except Exception as e:
        print(f"   ⚠ Erro ao gerar gráficos: {e}")
    
    print()
    
    # 8. Estatísticas finais
    print("8. Estatísticas Finais:")
    print("-" * 80)
    
    all_scores = [r.get('average_score', 0.0) for r in results]
    all_passed_list = [r.get('all_passed', False) for r in results]
    
    if all_scores:
        print(f"Score Médio Geral: {sum(all_scores) / len(all_scores):.3f}")
        print(f"Score Mínimo: {min(all_scores):.3f}")
        print(f"Score Máximo: {max(all_scores):.3f}")
        print(f"Taxa de Sucesso: {sum(all_passed_list) / len(all_passed_list):.1%}")
    
    # Estatísticas por métrica
    if results and 'metrics' in results[0]:
        print("\nEstatísticas por Métrica:")
        metric_stats = {}
        
        for result in results:
            if 'metrics' in result:
                for metric_name, metric_result in result['metrics'].items():
                    if metric_name not in metric_stats:
                        metric_stats[metric_name] = {'scores': [], 'successes': []}
                    
                    metric_stats[metric_name]['scores'].append(metric_result.get('score', 0.0))
                    metric_stats[metric_name]['successes'].append(metric_result.get('success', False))
        
        for metric_name, stats in metric_stats.items():
            avg_score = sum(stats['scores']) / len(stats['scores'])
            success_rate = sum(stats['successes']) / len(stats['successes'])
            print(f"  {metric_name}:")
            print(f"    Score Médio: {avg_score:.3f}")
            print(f"    Taxa de Sucesso: {success_rate:.1%}")
    
    print()
    print("=" * 80)
    print("AVALIAÇÃO CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    main()
