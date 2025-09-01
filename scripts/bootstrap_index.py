import argparse, os, glob, joblib
from genai_analytics.rag.retriever import TFIDFRetriever

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', type=str, default='data/docs')
    ap.add_argument('--out', type=str, default='configs/tfidf_index.joblib')
    args = ap.parse_args()
    retriever = TFIDFRetriever.fit_from_folder(args.docs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    retriever.save(args.out)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
