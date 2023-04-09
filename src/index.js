import hnswlib from "hnswlib-node";
import fs from "fs";
import duckdb from "duckdb";
import { table } from "console";

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

const createDatabaseAndIndex = async () => {
  const library = loadLibrary();

  const hnsw = new hnswlib.HierarchicalNSW("l2", DIMENSIONS);
  const numElements = library.bits.length;
  hnsw.initIndex(numElements);
  library.bits.forEach((bit, i) => {
    hnsw.addPoint(bit.embedding, i);
  });

  const db = new duckdb.Database("./store.db");
  const conn = db.connect();

  conn.exec("DROP TABLE IF EXISTS library");

  conn.exec(
    "CREATE TABLE library (id INTEGER, text VARCHAR, url VARCHAR, token_count INTEGER)"
  );

  const insertion = conn.prepare("INSERT INTO library VALUES (?, ?, ?, ?)");
  library.bits.forEach((bit, i) =>
    insertion.run(i, bit.text, bit.info.url, bit.token_count)
  );
  await new Promise((resolve, reject) => {
    insertion.finalize(resolve);
  });

  await hnsw.writeIndex("./index.bin");
};

const useIndex = async () => {
  const hnsw = new hnswlib.HierarchicalNSW("l2", DIMENSIONS);
  await hnsw.readIndex("./index.bin");

  const query = new Array(DIMENSIONS).fill(0).map(() => Math.random());
  const neighbors = 3;
  const results = hnsw.searchKnn(query, neighbors);

  console.log("Vector search results", results);

  const db = new duckdb.Database("./store.db");
  const conn = db.connect();

  conn.all("SELECT count(*) FROM library", (err, rows) => {
    console.log("Library size", err, rows);
  });

  conn.all(
    "SELECT * FROM library WHERE id IN (?, ?, ?)",
    ...results.neighbors,
    (err, rows) => {
      console.log("Text search results", err, rows);
    }
  );
};

const main = async () => {
  await createDatabaseAndIndex();
  await useIndex();
};

main();
