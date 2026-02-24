"""
Funções utilitárias para carregar dados, formatar resultados e gerar relatórios
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_llm_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Carrega dataset de teste para avaliação de LLM.
    
    Args:
        file_path: Caminho para o arquivo JSON com casos de teste
    
    Returns:
        Lista de dicts com casos de teste
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_agent_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Carrega dataset de teste para avaliação de agentes.
    
    Args:
        file_path: Caminho para o arquivo JSON com casos de teste de agentes
    
    Returns:
        Lista de dicts com casos de teste de agentes
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def format_results(
    results: List[Dict[str, Any]],
    output_format: str = 'dict'
) -> Any:
    """
    Formata resultados de avaliação em diferentes formatos.
    
    Args:
        results: Lista de resultados de avaliação
        output_format: Formato de saída ('dict', 'dataframe', 'json', 'summary')
    
    Returns:
        Resultados formatados conforme solicitado
    """
    if output_format == 'dict':
        return results
    
    elif output_format == 'dataframe':
        # Flatten results para DataFrame
        rows = []
        for result in results:
            row = {
                'test_id': result.get('test_case_id') or result.get('execution_id'),
                'input': result.get('input') or result.get('task', ''),
                'average_score': result.get('average_score', 0.0),
                'all_passed': result.get('all_passed', False)
            }
            
            # Adicionar scores individuais de métricas
            if 'metrics' in result:
                for metric_name, metric_result in result['metrics'].items():
                    row[f'{metric_name}_score'] = metric_result.get('score', 0.0)
                    row[f'{metric_name}_success'] = metric_result.get('success', False)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    elif output_format == 'json':
        return json.dumps(results, indent=2, ensure_ascii=False)
    
    elif output_format == 'summary':
        return _generate_summary(results)
    
    else:
        raise ValueError(f"Formato não suportado: {output_format}")


def _generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gera resumo estatístico dos resultados"""
    total_tests = len(results)
    
    if total_tests == 0:
        return {'error': 'Nenhum resultado para resumir'}
    
    # Coletar todas as métricas
    all_metrics = set()
    metric_scores = {}
    metric_successes = {}
    
    for result in results:
        if 'metrics' in result:
            for metric_name, metric_result in result['metrics'].items():
                all_metrics.add(metric_name)
                
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                    metric_successes[metric_name] = []
                
                metric_scores[metric_name].append(metric_result.get('score', 0.0))
                metric_successes[metric_name].append(metric_result.get('success', False))
    
    # Calcular estatísticas
    summary = {
        'total_tests': total_tests,
        'metrics': {}
    }
    
    for metric_name in all_metrics:
        scores = metric_scores[metric_name]
        successes = metric_successes[metric_name]
        
        summary['metrics'][metric_name] = {
            'average_score': sum(scores) / len(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'success_rate': sum(successes) / len(successes) if successes else 0.0,
            'total_evaluations': len(scores)
        }
    
    # Estatísticas gerais
    avg_scores = [r.get('average_score', 0.0) for r in results]
    all_passed = [r.get('all_passed', False) for r in results]
    
    summary['overall'] = {
        'average_score': sum(avg_scores) / len(avg_scores) if avg_scores else 0.0,
        'min_score': min(avg_scores) if avg_scores else 0.0,
        'max_score': max(avg_scores) if avg_scores else 0.0,
        'all_passed_rate': sum(all_passed) / len(all_passed) if all_passed else 0.0
    }
    
    return summary


def generate_report(
    results: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    include_details: bool = True
) -> str:
    """
    Gera relatório textual dos resultados de avaliação.
    
    Args:
        results: Lista de resultados de avaliação
        output_file: Caminho para salvar o relatório (opcional)
        include_details: Se True, inclui detalhes de cada caso de teste
    
    Returns:
        String com o relatório formatado
    """
    summary = _generate_summary(results)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RELATÓRIO DE AVALIAÇÃO")
    report_lines.append("=" * 80)
    report_lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total de Testes: {summary['total_tests']}")
    
    # Verificar se há erros gerais
    total_errors = sum(
        1 for result in results
        if 'metrics' in result
        for metric_result in result['metrics'].values()
        if 'error' in metric_result
    )
    if total_errors > 0:
        report_lines.append(f"⚠ Total de Erros Encontrados: {total_errors}")
    
    report_lines.append("")
    
    # Estatísticas gerais
    if 'overall' in summary:
        overall = summary['overall']
        report_lines.append("ESTATÍSTICAS GERAIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Score Médio: {overall['average_score']:.3f}")
        report_lines.append(f"Score Mínimo: {overall['min_score']:.3f}")
        report_lines.append(f"Score Máximo: {overall['max_score']:.3f}")
        report_lines.append(f"Taxa de Sucesso: {overall['all_passed_rate']:.1%}")
        report_lines.append("")
    
    # Estatísticas por métrica
    if 'metrics' in summary and summary['metrics']:
        report_lines.append("ESTATÍSTICAS POR MÉTRICA")
        report_lines.append("-" * 80)
        
        # Contar erros por métrica
        metric_errors = {}
        for result in results:
            if 'metrics' in result:
                for metric_name, metric_result in result['metrics'].items():
                    if 'error' in metric_result:
                        if metric_name not in metric_errors:
                            metric_errors[metric_name] = 0
                        metric_errors[metric_name] += 1
        
        for metric_name, stats in summary['metrics'].items():
            report_lines.append(f"\n{metric_name.upper()}:")
            report_lines.append(f"  Score Médio: {stats['average_score']:.3f}")
            report_lines.append(f"  Score Mínimo: {stats['min_score']:.3f}")
            report_lines.append(f"  Score Máximo: {stats['max_score']:.3f}")
            report_lines.append(f"  Taxa de Sucesso: {stats['success_rate']:.1%}")
            report_lines.append(f"  Total de Avaliações: {stats['total_evaluations']}")
            if metric_name in metric_errors:
                report_lines.append(f"  ⚠ Avaliações com Erro: {metric_errors[metric_name]}")
        
        report_lines.append("")
    
    # Detalhes de cada caso de teste
    if include_details:
        report_lines.append("DETALHES DOS CASOS DE TESTE")
        report_lines.append("-" * 80)
        
        for i, result in enumerate(results, 1):
            test_id = result.get('test_case_id') or result.get('execution_id') or i
            report_lines.append(f"\nTeste #{test_id}")
            
            if 'input' in result:
                report_lines.append(f"  Input: {result['input'][:100]}...")
            elif 'task' in result:
                report_lines.append(f"  Tarefa: {result['task'][:100]}...")
            
            report_lines.append(f"  Score Médio: {result.get('average_score', 0.0):.3f}")
            report_lines.append(f"  Todos Passaram: {'Sim' if result.get('all_passed') else 'Não'}")
            
            if 'metrics' in result:
                for metric_name, metric_result in result['metrics'].items():
                    score = metric_result.get('score', 0.0)
                    success = metric_result.get('success', False)
                    status = "✓" if success else "✗"
                    report_lines.append(f"    {status} {metric_name}: {score:.3f}")
                    
                    # Mostrar erro se houver
                    if 'error' in metric_result:
                        error_msg = metric_result['error']
                        # Truncar mensagem de erro muito longa
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200] + "..."
                        report_lines.append(f"      ⚠ Erro: {error_msg}")
                    
                    # Mostrar reason se score for 0 e houver reason
                    if score == 0.0 and 'reason' in metric_result:
                        reason = metric_result['reason']
                        if len(reason) > 150:
                            reason = reason[:150] + "..."
                        report_lines.append(f"      ℹ Razão: {reason}")
    
    report_text = "\n".join(report_lines)
    
    # Salvar em arquivo se especificado
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text


def plot_metrics(
    results: List[Dict[str, Any]],
    output_file: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Gera visualizações dos resultados de avaliação.
    
    Args:
        results: Lista de resultados de avaliação
        output_file: Caminho para salvar o gráfico (opcional)
        figsize: Tamanho da figura (largura, altura)
    """
    # Converter para DataFrame
    df = format_results(results, output_format='dataframe')
    
    if df.empty:
        print("Nenhum dado para visualizar")
        return
    
    # Preparar dados para visualização
    metric_columns = [col for col in df.columns if col.endswith('_score')]
    
    if not metric_columns:
        print("Nenhuma métrica encontrada para visualizar")
        return
    
    # Criar figura com subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfico 1: Distribuição de scores por métrica
    metric_data = []
    for col in metric_columns:
        metric_name = col.replace('_score', '')
        scores = df[col].dropna().tolist()
        for score in scores:
            metric_data.append({'Métrica': metric_name, 'Score': score})
    
    if metric_data:
        df_plot = pd.DataFrame(metric_data)
        sns.boxplot(data=df_plot, x='Métrica', y='Score', ax=axes[0])
        axes[0].set_title('Distribuição de Scores por Métrica')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Taxa de sucesso por métrica
    success_columns = [col for col in df.columns if col.endswith('_success')]
    if success_columns:
        success_data = []
        for col in success_columns:
            metric_name = col.replace('_success', '')
            success_rate = df[col].sum() / len(df) if len(df) > 0 else 0
            success_data.append({'Métrica': metric_name, 'Taxa de Sucesso': success_rate})
        
        if success_data:
            df_success = pd.DataFrame(success_data)
            sns.barplot(data=df_success, x='Métrica', y='Taxa de Sucesso', ax=axes[1])
            axes[1].set_title('Taxa de Sucesso por Métrica')
            axes[1].set_ylabel('Taxa de Sucesso')
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_file}")
    else:
        plt.show()
    
    plt.close()
