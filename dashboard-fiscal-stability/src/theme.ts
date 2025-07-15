import { extendTheme } from '@chakra-ui/react';

const colors = {
  brand: {
    900: '#1a202c', // azul-ônix
    800: '#2d3748',
    700: '#4a5568',
    600: '#718096',
    500: '#a0aec0',
    400: '#cbd5e0',
    300: '#e2e8f0',
    200: '#edf2f7',
    100: '#f7fafc',
    50: '#ffffff',
  },
  orange: {
    500: '#ff5722', // laranja vibrante
  },
  green: {
    500: '#8bc34a', // verde-limão
  },
};

const theme = extendTheme({ colors });

export default theme;
