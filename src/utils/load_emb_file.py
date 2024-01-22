# Load the spatial graph embedding information from saved .txt file

def load_emb_info(dir, name):
    emb_info_file = open(f'{dir}/{name}_emb_info.txt', 'r')
    _ = emb_info_file.readline()
    hyp_emb_line = emb_info_file.readline()
    _, hyp_dim, hyp_copy = hyp_emb_line.split()
    sph_emb_line = emb_info_file.readline()
    _, sph_dim, sph_copy = sph_emb_line.split()
    euc_emb_line = emb_info_file.readline()
    _, euc_dim, euc_copy = euc_emb_line.split()
    emb_info_file.close()
    
    print('Spatial embedding information successfully loaded.')
    
    return int(hyp_dim), int(hyp_copy), int(sph_dim), int(sph_copy), int(euc_dim), int(euc_copy)
