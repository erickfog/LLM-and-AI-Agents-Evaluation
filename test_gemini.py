#!/usr/bin/env python3
"""
Script de teste para verificar se o Gemini está funcionando corretamente
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()

from src.metrics_llm import GeminiJudgeModel, evaluate_answer_relevancy

def test_gemini_connection():
    """Testa a conexão com o Gemini"""
    print("=" * 80)
    print("TESTE DE CONEXÃO COM GEMINI")
    print("=" * 80)
    print()
    
    # Verificar API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERRO: GEMINI_API_KEY não encontrada no ambiente")
        print("Configure no arquivo .env:")
        print("GEMINI_API_KEY=sua_chave_aqui")
        return False
    
    print(f"✓ API Key encontrada: {api_key[:10]}...")
    print()
    
    # Testar GeminiJudgeModel
    try:
        print("Testando GeminiJudgeModel...")
        model = GeminiJudgeModel()
        print("✓ GeminiJudgeModel inicializado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao inicializar GeminiJudgeModel: {e}")
        return False
    
    # Testar geração simples
    try:
        print("\nTestando geração de resposta...")
        response = model.generate("Responda apenas com 'OK' se você está funcionando.")
        print(f"✓ Resposta recebida: {response[:100]}")
    except Exception as e:
        print(f"❌ Erro ao gerar resposta: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Testar métrica completa
    try:
        print("\nTestando métrica de Answer Relevancy...")
        result = evaluate_answer_relevancy(
            input_text="Qual é a capital do Brasil?",
            actual_output="A capital do Brasil é Brasília.",
            threshold=0.5
        )
        print(f"✓ Métrica executada:")
        print(f"  Score: {result.get('score', 'N/A')}")
        print(f"  Success: {result.get('success', 'N/A')}")
        print(f"  Reason: {result.get('reason', 'N/A')[:100]}...")
        if 'error' in result:
            print(f"  ⚠ Erro: {result['error']}")
    except Exception as e:
        print(f"❌ Erro ao executar métrica: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 80)
    print("✓ TODOS OS TESTES PASSARAM!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_gemini_connection()
    sys.exit(0 if success else 1)
