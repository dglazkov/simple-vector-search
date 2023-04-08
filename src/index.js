import hnswlib from "hnswlib-node";
import fs from "fs";

const DIMENSIONS = 1536;

const unpack = (packed) => {
  return Array.from(
    new Float32Array(new Uint8Array(Buffer.from(packed, "base64")).buffer)
  );
};

const loadLibrary = () => {
  const s = fs.readFileSync("./libraries/wdl-library.json");
  const library = JSON.parse(s);
  library.bits.forEach((bit) => {
    bit.embedding = unpack(bit.embedding);
  });
  return library;
};

const createIndex = async () => {
  const library = loadLibrary();

  const hnsw = new hnswlib.HierarchicalNSW("l2", DIMENSIONS);
  const numElements = library.bits.length;
  hnsw.initIndex(numElements);
  library.bits.forEach((bit, i) => {
    hnsw.addPoint(bit.embedding, i);
  });

  await hnsw.writeIndex("./index.bin");
};

const useIndex = () => {
  const hnsw = new hnswlib.HierarchicalNSW("l2", DIMENSIONS);
  hnsw.readIndexSync("./index.bin");

  const query = new Array(DIMENSIONS).fill(0).map(() => Math.random());
  const neighbors = 3;
  const results = hnsw.searchKnn(query, neighbors);
  console.log(results);
};

const main = async () => {
  await createIndex();
  useIndex();
};

main();
