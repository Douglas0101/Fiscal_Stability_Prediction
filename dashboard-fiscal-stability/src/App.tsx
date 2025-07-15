import { Routes, Route } from 'react-router-dom';
import { Box } from '@chakra-ui/react';
import Dashboard from './pages/Dashboard';

function App() {
  return (
    <Box>
      <Routes>
        <Route path="/" element={<Dashboard />} />
      </Routes>
    </Box>
  );
}

export default App;