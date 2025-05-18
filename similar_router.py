import xml.etree.ElementTree as ET
from collections import Counter
import sys # Importa o módulo sys para acessar os argumentos da linha de comando

def analyze_trips_from_xml(file_path):
    """
    Analisa um arquivo XML de viagens para contar pares origem-destino
    e calcular a economia potencial com cache, de forma eficiente em memória.
    """
    trip_counts = Counter()
    total_trips = 0

    try:
        print(f"Iniciando análise do arquivo: {file_path}")
        # Abre o arquivo e processa os elementos 'trip' conforme são encontrados
        # 'end' event é usado para processar o elemento quando ele está completo
        context = ET.iterparse(file_path, events=('end',))
        
        # Para obter o elemento raiz e limpá-lo no final se a estrutura for muito profunda
        # ou se o próprio elemento raiz retiver muita informação além dos filhos iterados.
        # No entanto, para a estrutura dada, limpar 'trip' é o mais crítico.
        # _, root = next(context) # Descomente se precisar de acesso ao root e para limpar depois

        for event, elem in context:
            if elem.tag == 'trip':
                origin = elem.get('origin')
                destination = elem.get('destination')
                
                if origin and destination: # Garante que os atributos existem
                    trip_counts[(origin, destination)] += 1
                    total_trips += 1
                
                # Crucial para economizar memória:
                # Limpa o elemento da memória depois de processado.
                elem.clear()
            
            # Se você descomentou a linha para obter 'root', e precisa liberar a árvore
            # gradualmente de baixo para cima, você pode adicionar lógica para limpar os pais
            # dos elementos 'trip' ou limpar o próprio 'root' no final do loop.
            # Para a estrutura <scsimulator_matrix><trip/>...</scsimulator_matrix>, limpar 'trip'
            # é geralmente suficiente, pois o iterador não mantém todos os 'trip' na memória simultaneamente.
            # Se 'scsimulator_matrix' em si fosse um elemento que acumulasse muitos atributos diretos
            # ou texto, e não apenas filhos 'trip', limpar o 'root' no final seria benéfico.
            # Exemplo:
            # if elem.tag == 'scsimulator_matrix': # Ou a tag raiz do seu XML
            #     elem.clear() # Isso limparia a raiz APÓS todos os filhos terem sido processados (se iterparse a mantiver)

        unique_trips = len(trip_counts)
        redundant_calculations = total_trips - unique_trips

        print(f"\nAnálise do arquivo '{file_path}' concluída:")
        print(f"Total de viagens (solicitações de cálculo de rota): {total_trips}")
        print(f"Número de pares únicos de origem-destino: {unique_trips}")
        print(f"Número de cálculos de rota que seriam economizados com cache: {redundant_calculations}\n")

        if total_trips == 0:
            print("Nenhuma viagem encontrada no arquivo.")
            return

        print("Contagem de viagens por par origem-destino (mostrando os mais comuns primeiro):")
        # Mostra os 10 mais comuns, ou todos se forem menos de 10, para não poluir a saída
        display_limit = 10
        common_trips = trip_counts.most_common(display_limit if len(trip_counts) > display_limit else len(trip_counts))
        
        for (origin, destination), count in common_trips:
            print(f"  Origem: {origin}, Destino: {destination} - Contagem: {count}")
            if count > 1:
                print(f"    -> Esta rota se repetiria {count - 1} vez(es) e poderia ser servida pelo cache.")
        if len(trip_counts) > display_limit:
            print(f"  ... e mais {len(trip_counts) - display_limit} outros pares únicos de origem-destino.")


        print("\nComo o Redis (ou outro sistema de cache) ajudaria:")
        print("1. Na primeira vez que uma rota (ex: N1 para N9) é solicitada, o cálculo é feito e o resultado é armazenado no cache (ex: Redis, com uma chave como 'route:N1:N9').")
        print("2. Em solicitações subsequentes para a MESMA rota (N1 para N9), o sistema primeiro verificaria o cache.")
        print("3. Se a chave existir no cache, o resultado da rota é retornado instantaneamente, evitando um novo cálculo computacionalmente intensivo.")
        if redundant_calculations > 0:
            print(f"4. Neste conjunto de dados, {redundant_calculations} cálculo(s) de rota poderiam ser evitados, pois são para pares origem-destino que já teriam sido calculados e cacheados.")
        else:
            print("4. Neste conjunto de dados, todas as rotas são únicas (ou não há rotas), portanto, não haveria economia de recálculo imediato com cache dentro deste lote específico de viagens. O cache ainda seria útil para solicitações futuras ou repetidas ao longo do tempo.")

    except FileNotFoundError:
        print(f"Erro: Arquivo '{file_path}' não encontrado.")
        sys.exit(1) # Termina o script com código de erro
    except ET.ParseError as e:
        print(f"Erro: O arquivo '{file_path}' não é um XML válido ou está corrompido. Detalhes: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o processamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Verifica se o caminho do arquivo foi fornecido como argumento
    if len(sys.argv) < 2:
        print("Erro: Caminho do arquivo XML não fornecido.")
        print("Uso: python nome_do_script.py <caminho_para_o_arquivo_xml>")
        # Exemplo de como criar um arquivo de teste para demonstração, se desejado.
        # print("\nCriando um arquivo de exemplo 'demo_trips.xml' para demonstração...")
        # sample_xml_content = """
        # <scsimulator_matrix>
        # <trip name="trip_1" origin="N1" destination="N9" link_origin="L1_2" count="1" start="0" mode="car" digital_rails_capable="false"/>
        # <trip name="trip_2" origin="N7" destination="N3" link_origin="L7_8" count="1" start="10" mode="car" digital_rails_capable="false"/>
        # <trip name="trip_3" origin="N1" destination="N9" link_origin="L10_6" count="1" start="20" mode="car" digital_rails_capable="false"/>
        # </scsimulator_matrix>
        # """
        # example_file_name = "demo_trips.xml"
        # with open(example_file_name, "w", encoding="utf-8") as f:
        #     f.write(sample_xml_content)
        # print(f"Para testar, use: python {sys.argv[0]} {example_file_name}")
        sys.exit(1) # Termina o script com código de erro

    # Pega o caminho do arquivo do primeiro argumento da linha de comando
    file_path_argument = sys.argv[1]
    
    # Chama a função principal de análise
    analyze_trips_from_xml(file_path_argument)