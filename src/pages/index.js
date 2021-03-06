import React from 'react'
import styled from 'styled-components'

import Layout from '../components/Layout'
import Page from '../components/Page'
import SocialMeta from '../components/SocialMeta'
import ProfilePic from '../components/ProfilePic'
import {
  Title,
  Section,
  Container,
  TextContainer,
  Flex,
  Col,
} from '../components/common'
import { isWhiteTheme } from '../utils'
import PostList from '../components/PostList'
import Wave from '../components/Wave'

const HeadSection = styled(Section)`
  padding-bottom: 0;
  padding-top: 7rem;
  position: relative;
`

export default ({ location }) => {
  const whiteTheme = isWhiteTheme({ location })

  return (
    <Layout location={location}>
      <Page>
        <SocialMeta />
        <HeadSection>
          <Container skinny>
            <Flex alignCenter>
              <Col>
                <ProfilePic whiteTheme={whiteTheme} size={125} />
                <Title>
                  <div className="background"/>
                  <span>Hamza Massaoudi</span>
                </Title>
                <TextContainer>
                  <p>
                    I am a Data Scientist living in Paris, France.
                  </p>
                  <p>
                    If you'd like to get in touch, send me an {' '}
                    <a
                      target="_blank"
                      rel="noopener noreferrer"
                      href="mailto:hamza.massaoudi.1994@gmail.com"
                    >
                      email
                    </a>{' '}
                    or find me on <a href="https://www.linkedin.com/in/hamza-massaoudi">Linkedin</a>{' '}
                    or <a href="https://twitter.com/HamzaMassaoudi">Twitter</a>
                  </p>
                </TextContainer>
              </Col>
            </Flex>
          </Container>
        </HeadSection>
      </Page>
      <Wave />
      <Page white>
        <PostList postCount={1} />
      </Page>
    </Layout>
  )
}
