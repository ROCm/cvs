import { Route, Routes } from "react-router-dom";
import SuiteList from "./SuiteList";
import SuiteDetail from "./SuiteDetail";

export default function CvsTile() {
  return (
    <Routes>
      <Route index element={<SuiteList />} />
      <Route path=":suite" element={<SuiteDetail />} />
    </Routes>
  );
}
