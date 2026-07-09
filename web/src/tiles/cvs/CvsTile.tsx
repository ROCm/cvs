import { Route, Routes } from "react-router-dom";
import SuiteList from "./SuiteList";
import SuiteDetail from "./SuiteDetail";
import ExecutionHistory from "./ExecutionHistory";
import ExecutionDetail from "./ExecutionDetail";

export default function CvsTile() {
  return (
    <Routes>
      <Route index element={<SuiteList />} />
      {/* Static routes before the dynamic :suite so they take precedence. */}
      <Route path="history" element={<ExecutionHistory />} />
      <Route path="executions/:id" element={<ExecutionDetail />} />
      <Route path=":suite" element={<SuiteDetail />} />
    </Routes>
  );
}
