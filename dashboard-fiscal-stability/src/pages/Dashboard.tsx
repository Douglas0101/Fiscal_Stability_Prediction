import { Box, Heading } from '@chakra-ui/react';
import LineChart from '../components/LineChart';

const Dashboard = () => {
  return (
    <Box p={8}>
      <Heading as="h1" mb={8}>Dashboard de Estabilidade Fiscal</Heading>
      <LineChart />
    </Box>
  );
};

export default Dashboard;
