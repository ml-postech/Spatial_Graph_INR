import os



class GraphAlreadyStartswithZeroError(Exception):
    pass

class GraphAlreadyStartswithOneError(Exception):
    pass


# Graph file format is assumed to be the list of edges
# ex)   0   1
#       0   2
#       ...        

# Receives a graph file to be converted for spatial embedding,
# so that the node index for the graph file starts with zero
def Graph_ZeroIndex_Start_Converter(graph):
    try:
        if os.path.isfile(graph) == False:
            raise FileNotFoundError
        
        dot_index = graph.find('.')
        ext_copy = graph[dot_index:]
        converted_graph = graph[:dot_index] + '_zero_start' + ext_copy
        
        src_file = open(graph, 'r')
        dest_file = open(converted_graph, 'w')
        
        while True:
            line = src_file.readline()
            
            if not line:
                break
            
            u, v = line.split(' ', 1)
            
            if (int(u) == 0 or int(v) == 0):
                raise GraphAlreadyStartswithZeroError(converted_graph)
            
            new_u = int(u) - 1
            new_v = int(v) - 1
            new_line = str(new_u) + ' ' + str(new_v) + '\n'
            dest_file.write(new_line)
        
        src_file.close()
        dest_file.close()
        
    except FileNotFoundError:
        print('Graph file was not found.')
    except GraphAlreadyStartswithZeroError as inst:
        new_file = inst.args[0]
        if os.path.isfile(new_file):
            os.remove(new_file)
        print('Graph was already modified for spatial embedding.')


# Receives a graph file to be converted for GINR,
# so that the node index for the graph file starts with one 
def Graph_OneIndex_Start_Converter(graph):
    try:
        if os.path.isfile(graph) == False:
            raise FileNotFoundError
        
        is_zero_found = False
        
        dot_index = graph.find('.')
        ext_copy = graph[dot_index:]
        converted_graph = graph[:dot_index] + '_one_start' + ext_copy
        
        src_file = open(graph, 'r')
        dest_file = open(converted_graph, 'w')
        
        while True:
            line = src_file.readline()
            
            if not line :
                break
            
            u, v = line.split(' ', 1)
            
            if (not is_zero_found) and ((int(u) == 0) or (int(v) == 0)):
                is_zero_found = True
            
            new_u = int(u) + 1
            new_v = int(v) + 1
            new_line = str(new_u) + ' ' + str(new_v) + '\n'
            dest_file.write(new_line)
            
            if not is_zero_found:
                raise GraphAlreadyStartswithOneError(converted_graph)
        
        src_file.close()
        dest_file.close()
        
    except FileNotFoundError:
        print('Graph file was not found.')
    except GraphAlreadyStartswithOneError as inst:
        new_file = inst.args[0]
        if(os.path.isfile(new_file)):
            os.remove(new_file)
        print('Graph was not need to be reverted.')
