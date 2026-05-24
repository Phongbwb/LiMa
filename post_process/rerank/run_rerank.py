import json
import post_process.rerank.rerank_attr
import post_process.rerank.intersection
import argparse

parser = argparse.ArgumentParser('rerank')
parser.add_argument('--rank_dir', type=str, default='./data/json/retrieval/baseline_rank.json', help='input directory of rank file')
parser.add_argument('--type_dir', type=str, default='./data/json/test_queries_type.json', help='input directory of type prediction file')
parser.add_argument('--color_dir', type=str, default='./data/json/test_queries_color.json', help='input directory of color prediction file')
parser.add_argument('--direction_dir', type=str, default='./data/json/test-tracks-direction-refinement.json', help='input directory of direction prediction file')
args = parser.parse_args()

rank_dir = args.rank_dir
type_dir = args.type_dir
colors_dir = args.color_dir
directions_dir = args.direction_dir

cur_rank = json.load(open(rank_dir))
type_dict = json.load(open(type_dir))
colors_dict = json.load(open(colors_dir))
directions_dir = json.load(open(directions_dir))

final_rank = post_process.rerank.intersection.rerank_intersection(post_process.rerank.rerank_attr.Rerank(cur_rank, type_dict, colors_dict, directions_dir))

json.dump(final_rank, open('./data/json/retrieval/final_rank.json', 'w'), indent=4)