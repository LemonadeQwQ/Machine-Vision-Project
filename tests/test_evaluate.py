"""Behavioral tests: known answers, ID alignment, invalid exports and CLI output."""
# 用途：用已知答案验证评估工具，防止编号错配、公式错误或非法输入产生不可靠分数。
# 这里不训练模型。多数测试临时创建 11 类人工标签和概率，另有一项读取真实验证标签。
# 每个 test_ 开头的方法都是一个独立测试；本文件目前包含 12 项测试。
# 在仓库根目录运行：python -m unittest discover -s tests -v。
import csv
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 让测试能导入仓库 src 目录中的评估模块，并共用同一份类别定义。
sys.path.insert(0, str(ROOT / "src"))
evaluation = importlib.import_module("evaluate")
CLASSES = evaluation.CLASSES


# unittest 会为每项测试准备独立实例；先运行 setUp，再运行对应测试方法。
class EvaluationTests(unittest.TestCase):
    # 每项测试都创建独立临时目录，结束后自动清理，避免改动项目中的正式数据和结果。
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vision-eval-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.labels = self.directory / "labels.csv"
        self.predictions = self.directory / "predictions.csv"
        # 默认样本：每个类别 1 个，共 11 个病变；用人工编号标明这是测试数据。
        self.label_rows = [{"lesion_id": f"synthetic_{i:02d}", "label": c}
                           for i, c in enumerate(CLASSES)]
        # 默认预测完全正确：真实类别概率为 1，其他类别为 0。
        # 后面的测试会复制或改写这份数据，构造乱序、错误预测和非法输入。
        self.prediction_rows = [
            {"lesion_id": row["lesion_id"], **{c: int(c == row["label"]) for c in CLASSES}}
            for row in self.label_rows
        ]
        self.write(self.labels, ("lesion_id", "label"), self.label_rows)
        self.write(self.predictions, ("lesion_id", *CLASSES), self.prediction_rows)

    # 测试辅助函数：按指定列顺序写入 CSV，用于主动制造不同输入情况。
    @staticmethod
    def write(path, columns, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    # 调用正式代码的读取入口，使测试覆盖实际使用时的数据检查和对齐过程。
    def load(self):
        return evaluation.load_evaluation_data(self.labels, self.predictions)

    # 测试 1：全部 11 类都预测正确时，整体指标应为 1，每类真实样本数应为 1。
    def test_perfect_predictions_for_all_classes(self):
        _, truth, probabilities = self.load()
        scores, per_class, matrix, normalized, _ = evaluation.calculate_metrics(truth, probabilities)
        self.assertEqual(scores, dict(accuracy=1.0, macro_f1=1.0, macro_precision=1.0, macro_recall=1.0))
        # np.eye(11) 是对角线为 1 的矩阵，恰好表示每类一个样本且全部分类正确。
        np.testing.assert_array_equal(matrix, np.eye(11))
        np.testing.assert_array_equal(normalized, np.eye(11))
        self.assertTrue(all(row["support"] == 1 for row in per_class))

    # 测试 2：同时反转行顺序和概率列顺序，结果仍应按编号和列名正确对齐。
    def test_rows_and_columns_match_by_name_not_position(self):
        _, expected_truth, expected_probabilities = self.load()
        columns = (*reversed(CLASSES), "lesion_id")
        self.write(self.predictions, columns, list(reversed(self.prediction_rows)))
        ids, truth, probabilities = self.load()
        self.assertEqual(ids, [r["lesion_id"] for r in self.label_rows])
        np.testing.assert_array_equal(truth, expected_truth)
        np.testing.assert_array_equal(probabilities, expected_probabilities)

    # 测试 3：构造可以手算的 4 个样本，同时验证缺失类别的处理。
    # 真实为 [BCC, BCC, NV, NV]，预测为 [BCC, NV, NV, NV]，所以 Accuracy=3/4。
    # BCC 的 Precision=1、Recall=1/2、F1=2/3；NV 的 F1=0.8。
    # 其余 9 类指标为 0，因此 Macro F1=(2/3+0.8)/11。
    def test_hand_calculated_errors_and_absent_classes(self):
        bcc, nv = CLASSES.index("BCC"), CLASSES.index("NV")
        truth = np.array([bcc, bcc, nv, nv])
        probabilities = np.eye(11)[[bcc, nv, nv, nv]]
        scores, rows, matrix, normalized, _ = evaluation.calculate_metrics(truth, probabilities)
        self.assertEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(rows[bcc]["precision"], 1.0)
        self.assertAlmostEqual(rows[bcc]["recall"], 0.5)
        self.assertAlmostEqual(rows[bcc]["f1"], 2 / 3)
        self.assertAlmostEqual(rows[nv]["precision"], 2 / 3)
        self.assertAlmostEqual(rows[nv]["recall"], 1.0)
        self.assertAlmostEqual(scores["macro_f1"], (2 / 3 + 0.8) / 11)
        self.assertEqual(matrix[bcc, nv], 1)
        self.assertEqual(normalized[bcc, nv], 0.5)
        self.assertEqual(rows[0]["support"], 0)
        np.testing.assert_array_equal(normalized[0], 0)

    # 测试 4：读取项目的真实验证标签，但人为将所有预测都设成 BCC。
    # 当前验证集有 1048 个病变，其中 BCC 有 504 个，所以准确率应为 504/1048。
    # BCC 的 F1=2*504/(1048+504)，其余类别 F1 为 0，宏平均再除以 11。
    # 这个参照用于验证计算，不代表任何训练模型的实际成绩。
    def test_real_validation_constant_bcc_reference(self):
        label_file = ROOT / "DATA/PROCESSED_DATA/val_split.csv"
        with label_file.open(encoding="utf-8-sig", newline="") as stream:
            labels = list(csv.DictReader(stream))
        predictions = [
            {"lesion_id": r["lesion_id"], **{c: int(c == "BCC") for c in CLASSES}}
            for r in reversed(labels)
        ]
        self.write(self.predictions, ("lesion_id", *CLASSES), predictions)
        _, truth, probabilities = evaluation.load_evaluation_data(label_file, self.predictions)
        scores, _, matrix, _, _ = evaluation.calculate_metrics(truth, probabilities)
        bcc_count = sum(row["label"] == "BCC" for row in labels)
        self.assertEqual(len(labels), 1048)
        self.assertEqual(bcc_count, 504)
        self.assertAlmostEqual(scores["accuracy"], 504 / 1048)
        self.assertAlmostEqual(scores["macro_f1"], (2 * 504 / (1048 + 504)) / 11)
        self.assertEqual(matrix[:, CLASSES.index("BCC")].sum(), 1048)

    # 测试 5：NaN、无穷大、负值、大于 1、空值、文本和概率和错误都应被拒绝。
    # 单独把正确类的概率从 1 改为 0.5，会使整行和变为 0.5，用来检查求和规则。
    def test_invalid_probability_values_and_sums_rejected(self):
        for value in ("nan", "inf", "-0.1", "1.1", "", "text", "0.5"):
            # subTest 让每个非法值都有独立的失败定位信息。
            with self.subTest(value=value):
                rows = [dict(r) for r in self.prediction_rows]
                rows[0][CLASSES[0]] = value
                self.write(self.predictions, ("lesion_id", *CLASSES), rows)
                with self.assertRaises(ValueError):
                    self.load()

    # 测试 6：依次构造缺一条、多一条、重复一条和编号带空格的预测文件。
    # 每种情况都应抛出错误，而不能通过忽略记录或自动修正编号继续计算。
    def test_missing_extra_duplicate_and_whitespace_ids_rejected(self):
        variants = [
            self.prediction_rows[:-1],
            self.prediction_rows + [{**self.prediction_rows[0], "lesion_id": "extra"}],
            self.prediction_rows + [self.prediction_rows[0]],
            [{**self.prediction_rows[0], "lesion_id": " synthetic_00"}, *self.prediction_rows[1:]],
        ]
        for rows in variants:
            with self.subTest(length=len(rows)):
                self.write(self.predictions, ("lesion_id", *CLASSES), rows)
                with self.assertRaises(ValueError):
                    self.load()

    # 测试 7：真实标签一侧也必须检查，拒绝重复编号、未知类别和不完整的 one-hot 列。
    def test_duplicate_true_ids_unknown_label_and_partial_one_hot_rejected(self):
        variants = [
            (("lesion_id", "label"), self.label_rows + [self.label_rows[0]]),
            (("lesion_id", "label"), [{**self.label_rows[0], "label": "UNKNOWN"}]),
            (("lesion_id", "label", "BCC"), [{**self.label_rows[0], "BCC": 0}]),
        ]
        for columns, rows in variants:
            with self.subTest(columns=columns):
                self.write(self.labels, columns, rows)
                with self.assertRaises(ValueError):
                    self.load()

    # 测试 8：第一条标签是 AKIEC，刻意让 one-hot 指向 BCC，应发现两者矛盾。
    def test_mismatched_one_hot_rejected(self):
        rows = [{**self.label_rows[0], **{c: int(c == "BCC") for c in CLASSES}}]
        self.write(self.labels, ("lesion_id", *CLASSES, "label"), rows)
        with self.assertRaisesRegex(ValueError, "one-hot"):
            self.load()

    # 测试 9：分别检查重复列名、只有表头、字段太少、字段太多的 CSV。
    def test_malformed_prediction_csv_rejected(self):
        for text in (
            "lesion_id,lesion_id\na,b\n",
            "lesion_id," + ",".join(CLASSES) + "\n",
            "lesion_id," + ",".join(CLASSES) + "\nx,1\n",
            "lesion_id," + ",".join(CLASSES) + "\nx," + ",".join(["0"] * 12) + "\n",
        ):
            with self.subTest(text=text[:30]):
                self.predictions.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()

    # 测试 10：0.999999 与 1 的差值在容差内，应被接受。
    # 另一个样本让 AKIEC 与 BCC 都为 0.5，最高概率并列时应选索引靠前的 AKIEC。
    def test_probability_tolerance_and_tie_order(self):
        rows = [dict(r) for r in self.prediction_rows]
        rows[0][CLASSES[0]] = 0.999999
        rows[1][CLASSES[0]], rows[1][CLASSES[1]] = 0.5, 0.5
        self.write(self.predictions, ("lesion_id", *CLASSES), rows)
        _, truth, probabilities = self.load()
        *_, predicted = evaluation.calculate_metrics(truth, probabilities)
        self.assertEqual(predicted[1], 0)

    # 测试 11：启动真实命令行进程，验证文件输出、跨目录调用及覆盖开关。
    # sys.executable 使用当前测试的 Python 解释器，避免误用其他已安装的 Python。
    def test_cli_from_another_directory_and_safe_rerun(self):
        out = self.directory / "results"
        command = [sys.executable, str(ROOT / "src/evaluate.py"),
                   "--labels", str(self.labels), "--predictions", str(self.predictions),
                   "--output-dir", str(out), "--model-name", "Synthetic perfect test"]
        # 第一次运行：在临时目录执行，检查退出码、七个输出文件和整体分数。
        first = subprocess.run(command, cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual({p.name for p in out.iterdir()}, set(evaluation.OUTPUT_FILES))
        summary = json.loads((out / "metrics.json").read_text())
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["macro_f1"], 1.0)
        self.assertEqual(summary["n_lesions"], 11)
        # 检查 PNG 文件头，确认两个图片输出确实采用 PNG 格式。
        for filename in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            self.assertEqual((out / filename).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        # 第二次运行未允许覆盖，应失败，并且原有 metrics.json 内容保持不变。
        second = subprocess.run(command, cwd=self.directory, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("--overwrite", second.stderr)
        self.assertEqual(summary, json.loads((out / "metrics.json").read_text()))
        # 第三次明确加上 --overwrite，应允许重新生成结果并正常退出。
        third = subprocess.run([*command, "--overwrite"], cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(third.returncode, 0, third.stderr)

    # 测试 12：把输入预测文件取名为某个固定输出文件名，制造路径冲突。
    # 即使允许覆盖旧结果，也不能覆盖输入数据；同时核对输入字节完全未变。
    def test_output_cannot_replace_input_even_with_overwrite(self):
        renamed = self.directory / "predictions_aligned.csv"
        self.predictions.rename(renamed)
        original = renamed.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            evaluation.evaluate_files(self.labels, renamed, self.directory, overwrite=True)
        self.assertEqual(renamed.read_bytes(), original)


# 直接运行此文件时也可以启动测试；unittest discover 会自动收集上面的测试方法。
if __name__ == "__main__":
    unittest.main()
