"""
Exemplo completo de avaliação de LLM usando múltiplas métricas
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics_llm import evaluate_llm_batch
from src.utils import load_llm_dataset, format_results, generate_report, plot_metrics


def main():
    """Executa avaliação completa de LLM"""
    
    print("=" * 80)
    print("EXEMPLO: AVALIAÇÃO DE LLM COM MÚLTIPLAS MÉTRICAS")
    print("=" * 80)
    print()
    
    # 1. Carregar dataset
    print("1. Carregando dataset de teste...")
    dataset_path = Path(__file__).parent.parent / "data" / "test_dataset_llm.json"
    
    try:
        test_cases = load_llm_dataset(str(dataset_path))
        print(f"   ✓ Carregados {len(test_cases)} casos de teste")
    except Exception as e:
        print(f"   ✗ Erro ao carregar dataset: {e}")
        return
    
    print()
    
    # 2. Preparar casos de teste no formato esperado
    print("2. Preparando casos de teste...")
    formatted_cases = []
    
    for case in test_cases:
        formatted_case = {
            'input': case.get('input', ''),
            'actual_output': case.get('actual_output', ''),
            'expected_output': case.get('expected_output'),
            'context': case.get('context')
        }
        
        # Adicionar critério para G-Eval se disponível
        if 'criteria' in case:
            formatted_case['criteria'] = case['criteria']
        
        formatted_cases.append(formatted_case)
    
    print(f"   ✓ {len(formatted_cases)} casos formatados")
    print()
    
    # 3. Definir métricas a avaliar
    print("3. Configurando métricas...")
    metrics = ['relevancy', 'faithfulness']
    
    # Adicionar bias_toxicity se não houver contexto (métrica referenceless)
    metrics.append('bias_toxicity')
    
    # Adicionar geval se houver critérios customizados
    if any('criteria' in case for case in formatted_cases):
        metrics.append('geval')
    
    print(f"   ✓ Métricas selecionadas: {', '.join(metrics)}")
    print()
    
    # 4. Executar avaliações
    print("4. Executando avaliações (isso pode levar alguns minutos)...")
    print("   Aguarde enquanto as métricas são calculadas...")
    print()
    
    try:
        results = evaluate_llm_batch(
            test_cases=formatted_cases,
            metrics=metrics,
            thresholds={
                'relevancy': 0.5,
                'faithfulness': 0.5,
                'bias_toxicity': 0.7,
                'geval': 0.5
            }
        )
        print(f"   ✓ Avaliação concluída para {len(results)} casos")
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
        test_id = result.get('test_case_id', i)
        avg_score = result.get('average_score', 0.0)
        all_passed = result.get('all_passed', False)
        
        print(f"\nTeste #{test_id}:")
        print(f"  Score Médio: {avg_score:.3f}")
        print(f"  Status: {'✓ PASSOU' if all_passed else '✗ FALHOU'}")
        
        if 'metrics' in result:
            for metric_name, metric_result in result['metrics'].items():
                score = metric_result.get('score', 0.0)
                success = metric_result.get('success', False)
                status = "✓" if success else "✗"
                print(f"    {status} {metric_name}: {score:.3f}")
                
                # Mostrar erro se houver
                if 'error' in metric_result:
                    error_msg = metric_result['error']
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    print(f"      ⚠ Erro: {error_msg}")
    
    print()
    
    # 6. Gerar relatório
    print("6. Gerando relatório...")
    report_path = Path(__file__).parent.parent / "reports" / "llm_evaluation_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    report = generate_report(results, output_file=str(report_path), include_details=True)
    print(f"   ✓ Relatório salvo em: {report_path}")
    print()
    
    # 7. Gerar visualizações
    print("7. Gerando visualizações...")
    plot_path = Path(__file__).parent.parent / "reports" / "llm_evaluation_plots.png"
    
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
    
    print()
    print("=" * 80)
    print("AVALIAÇÃO CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    main()
