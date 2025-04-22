require('dotenv').config();

if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL is not set in .env');
}

module.exports = {
  dialect: 'postgresql',
  dbCredentials: {
    url: process.env.DATABASE_URL,
  },
  tablesFilter: ['front-end_*'],
};
