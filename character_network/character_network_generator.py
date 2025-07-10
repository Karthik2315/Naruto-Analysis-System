import pandas as pd
from pyvis.network import Network
import networkx as nx
import html

class CharacterNetworkGenerator():
  def __init__(self):
    pass
  
  def generate_character_network(self,df):
    windows = 10
    entity_relationship = []
    
    for row in df['ners']:
      previous_entities_in_windows = []
      for sentence in row:
        previous_entities_in_windows.extend(list(sentence))
        previous_entities_in_windows = previous_entities_in_windows[-windows:]
        for entity in sentence:
          for entity_in_windows in previous_entities_in_windows:
            if entity != entity_in_windows:
              entity_relationship.append(sorted([entity,entity_in_windows]))
    relational_df = pd.DataFrame({'values':entity_relationship})
    relational_df['source'] = relational_df['values'].apply(lambda x:x[0])
    relational_df['target'] = relational_df['values'].apply(lambda x:x[1])
    relational_df = relational_df.groupby(['source','target']).size().reset_index(name="weight").sort_values(by='weight',inplace=False,ascending=False)
    return relational_df
  
  def draw_network_graph(self,df):
    df = df.head(200)
    G = nx.from_pandas_edgelist(
        df,
        source='source',
        target='target',
        edge_attr='weight',
        create_using=nx.Graph()
    )
    net = Network(
        notebook=True,
        height="700px",
        width="1000px",
        bgcolor="#222222",
        font_color="white", 
        cdn_resources="remote"
    )
    node_degree = dict(G.degree)
    for node, degree in node_degree.items():
      if degree>50:
          color = "#ff6666"
      elif degree>10:
          color = "#ff7f7f"
      else:
          color = "orange"
      net.add_node(node, size=degree * 1,color=color)  

    for source, target, data in G.edges(data=True):
      weight = data.get('weight',1)
      if weight > 50:
          color = "#ff6666"
      elif weight >10:
          color = "#ff7f7f"
      else:
          color = "orange"
      net.add_edge(source, target, value=data.get('weight', 1),color=color)  
    
    raw_html = net.generate_html()
    #html = html.replace("'","\'")
    escaped_html = html.escape(raw_html)
    output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; display-capture; encrypted-media;" sandbox="allow-modals allow-forms allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" allowpaymentrequest="" frameborder="0" srcdoc='{escaped_html}'></iframe>"""
    
    return output_html

    